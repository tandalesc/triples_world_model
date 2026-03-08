"""T5-based text decoder for Triple World Model value generation.

Uses a frozen pretrained T5 decoder to generate free-text value phrases
from TWM latent embeddings. A lightweight projection layer maps the full
triple context (entity + attr + value positions) into a multi-token
sequence in T5's cross-attention space.

Entity and attr positions use discrete vocab heads (small, closed vocab).
Value positions are decoded by T5 into open-ended text.
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5ValueDecoder(nn.Module):
    """Projects TWM latent vectors into T5 decoder space for text generation.

    Architecture:
        Triple context (entity + attr + value embeddings, 3*d_model)
        → MLP → (n_proj_tokens * t5_dim)
        → reshape to (n_proj_tokens, t5_dim)
        → Frozen T5 decoder cross-attends to multi-token sequence
        → token logits → text

    Each projected token can specialize: semantic category, specificity,
    sentiment, etc. This gives T5 multiple angles to attend to instead
    of trying to reconstruct a phrase from a single key.
    """

    def __init__(
        self,
        twm_dim: int = 256,
        t5_model_name: str = "t5-small",
        n_proj_tokens: int = 8,
        unfreeze_last_n: int = 0,
    ):
        super().__init__()
        self.twm_dim = twm_dim
        self.t5_model_name = t5_model_name
        self.n_proj_tokens = n_proj_tokens

        # Load T5 — we only need the decoder + lm_head
        t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.t5_config = t5.config
        self.t5_dim = t5.config.d_model  # 512 for t5-small

        # Extract decoder and lm_head
        self.t5_decoder = t5.decoder
        self.lm_head = t5.lm_head

        # Freeze everything by default
        for param in self.t5_decoder.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

        # Optionally unfreeze last N decoder layers
        if unfreeze_last_n > 0:
            layers = self.t5_decoder.block
            for layer in layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Projection: full triple context (3*d_model) → n_proj_tokens * t5_dim
        # Input is concatenated entity+attr+value from dynamics output
        input_dim = twm_dim * 3  # entity + attr + value
        output_dim = n_proj_tokens * self.t5_dim
        hidden_dim = max(input_dim, output_dim // 2)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    def project(self, triple_context: torch.Tensor) -> torch.Tensor:
        """Project triple context to multi-token T5 cross-attention sequence.

        Args:
            triple_context: (B, 3*twm_dim) concatenated entity+attr+value

        Returns:
            (B, n_proj_tokens, t5_dim) multi-token sequence for cross-attention
        """
        B = triple_context.shape[0]
        flat = self.projection(triple_context)  # (B, n_proj_tokens * t5_dim)
        return flat.view(B, self.n_proj_tokens, self.t5_dim)

    def forward(
        self,
        triple_context: torch.Tensor,
        target_ids: torch.Tensor,
        target_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass for training.

        Args:
            triple_context: (B, 3*twm_dim) concatenated entity+attr+value embeddings
            target_ids: (B, seq_len) tokenized target value phrases
            target_attention_mask: (B, seq_len) attention mask for targets

        Returns:
            logits: (B, seq_len, vocab_size) token logits
        """
        encoder_hidden = self.project(triple_context)  # (B, K, t5_dim)
        K = self.n_proj_tokens

        decoder_out = self.t5_decoder(
            input_ids=target_ids,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=torch.ones(
                encoder_hidden.shape[0], K,
                device=encoder_hidden.device, dtype=torch.long,
            ),
        )

        return self.lm_head(decoder_out.last_hidden_state)

    @torch.no_grad()
    def generate(
        self,
        triple_context: torch.Tensor,
        max_length: int = 32,
    ) -> list[str]:
        """Generate value phrases from triple context.

        Args:
            triple_context: (B, 3*twm_dim) concatenated entity+attr+value

        Returns:
            list of B decoded strings
        """
        encoder_hidden = self.project(triple_context)
        B = encoder_hidden.shape[0]
        K = self.n_proj_tokens

        start_id = self.t5_config.decoder_start_token_id
        input_ids = torch.full(
            (B, 1), start_id,
            dtype=torch.long, device=encoder_hidden.device,
        )
        encoder_mask = torch.ones(B, K, device=encoder_hidden.device, dtype=torch.long)
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        finished = torch.zeros(B, dtype=torch.bool, device=encoder_hidden.device)

        for _ in range(max_length):
            decoder_out = self.t5_decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=encoder_mask,
            )
            next_logits = self.lm_head(decoder_out.last_hidden_state[:, -1:])
            next_id = next_logits.argmax(dim=-1)  # (B, 1)

            # Freeze finished sequences — replace with pad so no post-EOS garbage
            finished = finished | (next_id.squeeze(-1) == eos_id)
            next_id[finished] = pad_id
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if finished.all():
                break

        texts = self.tokenizer.batch_decode(input_ids[:, 1:], skip_special_tokens=True)
        return texts

    def frozen_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

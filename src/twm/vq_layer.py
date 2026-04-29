"""Vector Quantization layer (VQ-VAE style).

Quantizes continuous vectors to nearest entry in a learned codebook.
Forward returns the quantized vector via straight-through estimator
so the encoder still receives gradient. Backward updates the codebook
through the codebook loss and pulls the encoder toward codes via the
commitment loss.

Used to force discrete entity IDs in the dynamics output, where the
diagnosis says "red potato" and "red hot pepper" collapse together
in continuous space. With a discrete codebook of size N >> num_entities,
similar-but-distinct entities can snap to different codes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, d: int, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.d = d
        self.beta = beta

        # Init to match LayerNorm-style encoder outputs (unit per-element variance).
        # The original VQ-VAE init uniform_(-1/N, 1/N) assumes encoder magnitudes
        # of ~1/N, which collapses immediately when fed O(1) bottleneck values.
        self.codebook = nn.Embedding(num_codes, d)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Quantize z.

        Args:
            z: (..., d) input vectors

        Returns:
            z_q: (..., d) quantized vectors with straight-through gradient
            vq_loss: scalar codebook + commitment loss
            stats: dict {vq_loss, perplexity, unique_codes}
        """
        orig_shape = z.shape
        # Run distance / loss math in fp32 to avoid bf16 overflow on the
        # ||z||² + ||e||² - 2 z·e expansion (each term can be O(d) before sum).
        with torch.amp.autocast(device_type=z.device.type, enabled=False):
            z_flat = z.reshape(-1, self.d).float()
            codebook = self.codebook.weight.float()

            # ||z - e||² = ||z||² + ||e||² - 2 z·e
            d2 = (
                z_flat.pow(2).sum(-1, keepdim=True)
                + codebook.pow(2).sum(-1)
                - 2 * z_flat @ codebook.T
            )

            idx = d2.argmin(-1)
            z_q_flat = F.embedding(idx, codebook)

            codebook_loss = F.mse_loss(z_q_flat, z_flat.detach())
            commit_loss = F.mse_loss(z_flat, z_q_flat.detach())
            vq_loss = codebook_loss + self.beta * commit_loss

            # Straight-through: forward = z_q, gradient = z
            z_q_flat = z_flat + (z_q_flat - z_flat).detach()
            z_q = z_q_flat.reshape(orig_shape).to(z.dtype)

            with torch.no_grad():
                counts = torch.bincount(idx, minlength=self.num_codes).float()
                probs = counts / counts.sum()
                entropy = -(probs * (probs + 1e-10).log()).sum()
                stats = {
                    "vq_loss": vq_loss.item(),
                    "perplexity": entropy.exp().item(),
                    "unique_codes": (counts > 0).sum().item(),
                }

        return z_q, vq_loss, stats

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

        self.codebook = nn.Embedding(num_codes, d)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

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
        z_flat = z.reshape(-1, self.d)

        # ||z - e||² = ||z||² + ||e||² - 2 z·e
        d2 = (
            z_flat.pow(2).sum(-1, keepdim=True)
            + self.codebook.weight.pow(2).sum(-1)
            - 2 * z_flat @ self.codebook.weight.T
        )

        idx = d2.argmin(-1)
        z_q_flat = self.codebook(idx)

        codebook_loss = F.mse_loss(z_q_flat, z_flat.detach())
        commit_loss = F.mse_loss(z_flat, z_q_flat.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # Straight-through: forward = z_q, gradient = z
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()
        z_q = z_q_flat.reshape(orig_shape)

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

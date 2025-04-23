import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# ─── Utility functions to split & reassemble into patches ───────────────────────
def extract_patches(x: torch.Tensor, patch_size: int = 16):
    """
    x: (B, C, H, W)
    returns: (B, num_patches, C * patch_size * patch_size)
    """
    bs, c, h, w = x.size()
    unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    patches = unfold(x)               # (B, C*ps*ps, L)
    return patches.transpose(1, 2)    # (B, L, C*ps*ps)

def reconstruct_image(patch_seq: torch.Tensor, image_shape: tuple, patch_size: int = 16):
    """
    patch_seq: (B, L, C*ps*ps)
    image_shape: (B, C, H, W)
    """
    bs, c, h, w = image_shape
    num_h = h // patch_size
    num_w = w // patch_size

    # reshape back to (B, C*ps*ps, L)
    patches = patch_seq.transpose(1, 2)
    # fold back
    fold = nn.Fold(output_size=(h, w),
                   kernel_size=patch_size,
                   stride=patch_size)
    return fold(patches)  # (B, C, H, W)

# ─── Conditional LayerNorm for conditioning on timestep or other features ───────
class ConditionalNorm2d(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_scale = nn.Linear(cond_dim, dim)
        self.to_shift = nn.Linear(cond_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        # x: (B, L, D), cond: (B, cond_dim)
        x_norm = self.layernorm(x)
        scale = self.to_scale(cond).unsqueeze(1)  # (B,1,D)
        shift = self.to_shift(cond).unsqueeze(1)
        return x_norm * (1 + scale) + shift

# ─── Sinusoidal positional / timestep embedding ─────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        # t: (B,)  — timesteps
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)

# ─── A single Transformer block with conditional norm ──────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.condnorm = ConditionalNorm2d(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        # x: (B, L, D), cond: (B, cond_dim)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.condnorm(x, cond)
        h = self.mlp(h)
        return x + h

# ─── DiT: Diffusion Transformer Encoder ────────────────────────────────────────
class DiT(nn.Module):
    def __init__(
        self,
        image_size: int,
        channels_in: int,
        patch_size: int = 16,
        hidden_size: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        cond_dim: int = 128
    ):
        """
        image_size: H == W of input
        channels_in: e.g., 4 for VAE latents
        """
        super().__init__()
        self.patch_size = patch_size
        self.dim = hidden_size

        # time / step embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim * 2),
            nn.GELU(),
            nn.Linear(cond_dim * 2, cond_dim)
        )

        # project each patch to hidden dimension
        patch_dim = channels_in * patch_size * patch_size
        self.to_hidden = nn.Linear(patch_dim, hidden_size)

        # learnable positional embeddings for each patch
        num_patches = (image_size // patch_size) ** 2
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, hidden_size))

        # transformer layers
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, cond_dim)
            for _ in range(num_layers)
        ])

        # final projection back to patch_dim
        self.to_patch = nn.Linear(hidden_size, patch_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # x: (B, C, H, W), t: (B,) timesteps
        B, C, H, W = x.shape
        cond = self.time_mlp(t)             # (B, cond_dim)

        # extract & embed patches
        patches = extract_patches(x, self.patch_size)  # (B, L, patch_dim)
        h = self.to_hidden(patches)                   # (B, L, hidden)

        # add positional embeddings
        h = h + self.pos_emb

        # pass through transformer blocks
        for blk in self.blocks:
            h = blk(h, cond)

        # project back to pixels and reassemble
        out_patches = self.to_patch(h)                # (B, L, patch_dim)
        return reconstruct_image(out_patches, x.shape, self.patch_size)

# ─── Cosine schedule for diffusion alphas ───────────────────────────────────────
def cosine_alphas_bar(timesteps: int, s: float = 0.008):
    """
    Returns a (timesteps,) tensor of cumulative product of alphas
    following a cosine schedule.
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cum = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cum = alphas_cum / alphas_cum[0]
    return alphas_cum[:-1]

# ─── Dataset for loading precomputed latents ────────────────────────────────
class LatentDataset(Dataset):
    """Loads precomputed latents from .npy files."""
    def __init__(self, folder: Path):
        self.files = sorted(folder.glob("*.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        return torch.from_numpy(arr).float()
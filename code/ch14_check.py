"""Chapter 14 · Multimodal models: runnable checks (CPU, seconds)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# 14.2 ViT patchify: an image becomes a sequence of patch tokens
def patchify(img, p=16):                       # img: (C, H, W)
    C, H, W = img.shape
    x = img.unfold(1, p, p).unfold(2, p, p)    # (C, H/p, W/p, p, p)
    return x.permute(1, 2, 0, 3, 4).reshape(-1, C * p * p)
patches = patchify(torch.randn(3, 224, 224))
print("224x224 image, 16x16 patches:", tuple(patches.shape))
for side in (224, 448, 896, 1344):
    print(f"  {side}x{side}: {(side // 14) ** 2:>5} tokens with 14-px patches; "
          f"{(side // 28) ** 2:>5} after 2x2 merging")

# 14.2 CLIP (softmax) vs SigLIP (sigmoid) contrastive losses on a toy batch
B, d = 8, 32
img = F.normalize(torch.randn(B, d), dim=-1)
txt = F.normalize(img + 0.3 * torch.randn(B, d), dim=-1)      # matched pairs are similar
logits = img @ txt.T / 0.07
clip = (F.cross_entropy(logits, torch.arange(B)) + F.cross_entropy(logits.T, torch.arange(B))) / 2
t, b = 10.0, -10.0
labels = 2 * torch.eye(B) - 1                                  # +1 on the diagonal, -1 elsewhere
siglip = -F.logsigmoid(labels * (t * img @ txt.T + b)).sum() / B
print(f"CLIP loss {clip.item():.3f}  SigLIP loss {siglip.item():.3f}")

# 14.3 LLaVA-style assembly: vision features -> projector -> spliced into the text sequence
d_vis, d_llm, n_img_tokens = 1024, 2048, 576
projector = nn.Sequential(nn.Linear(d_vis, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))
vis = torch.randn(1, n_img_tokens, d_vis)
img_emb = projector(vis)
text_before, text_after = torch.randn(1, 12, d_llm), torch.randn(1, 20, d_llm)
seq = torch.cat([text_before, img_emb, text_after], dim=1)
labels = torch.full((1, seq.shape[1]), -100)
labels[0, 12 + n_img_tokens + 5:] = 1                          # loss only on the answer tokens
print("sequence", tuple(seq.shape), " tokens scored:", int((labels != -100).sum()),
      " projector params: %.1fM" % (sum(p.numel() for p in projector.parameters()) / 1e6))

# 14.3 pixel-shuffle token merging: 2x2 neighbouring tokens -> one token with 4x the channels
grid = torch.randn(1, 24, 24, d_vis)
merged = grid.reshape(1, 12, 2, 12, 2, d_vis).permute(0, 1, 3, 2, 4, 5).reshape(1, 144, 4 * d_vis)
print("pixel shuffle:", tuple(grid.reshape(1, -1, d_vis).shape), "->", tuple(merged.shape))

# 14.4 2-D rotary positions for image tokens: rotate half the dims by row, half by column
def rope_1d(x, pos, base=10000.0):
    half = x.shape[-1] // 2
    freqs = base ** (-torch.arange(half) / half)
    ang = pos[:, None] * freqs[None]
    return torch.cat([x[..., :half] * ang.cos() - x[..., half:] * ang.sin(),
                      x[..., :half] * ang.sin() + x[..., half:] * ang.cos()], -1)
def rope_2d(x, rows, cols):
    h = x.shape[-1] // 2
    return torch.cat([rope_1d(x[..., :h], rows), rope_1d(x[..., h:], cols)], -1)
q = torch.randn(1, 16).repeat(4, 1); k = torch.randn(1, 16).repeat(4, 1)
rows = torch.tensor([0., 0., 3., 3.]); cols = torch.tensor([0., 2., 5., 7.])
rq, rk = rope_2d(q, rows, cols), rope_2d(k, rows, cols)
# same (row, col) offset -> same score: (0,0)->(0,2) and (3,5)->(3,7)
print("2-D RoPE relative: %.4f %.4f" % ((rq[0] @ rk[1]).item(), (rq[2] @ rk[3]).item()))

# 14.5 Whisper-style audio token count
sr, hop, seconds = 16000, 160, 30
frames = seconds * sr // hop
print(f"30 s audio: {frames} mel frames -> {frames // 2} encoder tokens after a stride-2 conv")
# neural audio codec tokens: frames per second x codebooks
print("codec at 12.5 frames/s with 8 codebooks: %d tokens for 30 s" % (12.5 * 8 * 30))

# 14.6 vector quantisation (image or audio tokenizer): nearest codebook entry + straight-through gradient
codebook = torch.randn(512, 64)
z = torch.randn(10, 64, requires_grad=True)
idx = torch.cdist(z, codebook).argmin(-1)
zq = codebook[idx]
z_st = z + (zq - z).detach()                  # forward uses zq, backward passes gradients to z
commit = F.mse_loss(z, zq.detach())
print("VQ codes:", idx.tolist()[:6], " commitment loss %.3f" % commit.item())

# 14.6 diffusion forward process and the epsilon-prediction target
T_steps = 1000
betas = torch.linspace(1e-4, 0.02, T_steps)
alpha_bar = torch.cumprod(1 - betas, 0)
for t_ in (0, 250, 500, 999):
    print(f"t={t_:>3}: signal weight sqrt(a_bar)={alpha_bar[t_].sqrt():.3f}  noise weight={(1-alpha_bar[t_]).sqrt():.3f}")
x0 = torch.randn(4, 3, 8, 8); eps = torch.randn_like(x0); t_ = torch.tensor([10, 200, 600, 990])
xt = alpha_bar[t_].sqrt()[:, None, None, None] * x0 + (1 - alpha_bar[t_]).sqrt()[:, None, None, None] * eps
# a model predicts eps from (xt, t); classifier-free guidance mixes conditional and unconditional predictions
eps_c, eps_u, w = torch.randn_like(x0), torch.randn_like(x0), 5.0
eps_guided = eps_u + w * (eps_c - eps_u)
print("x_t std per sample:", [round(v, 2) for v in xt.flatten(1).std(1).tolist()])

# 14.7 video token budget
fps, minutes, per_frame = 2, 10, 256
print(f"{minutes} min video at {fps} fps, {per_frame} tokens/frame: {fps*60*minutes*per_frame:,} tokens")

"""Chapter 14 · Multimodal models: runnable checks (CPU, seconds)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)

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

# ---------------------------------------------------------------------------
# Additions (review pass)
# ---------------------------------------------------------------------------

# 14.5 a complete VQ objective: the codebook needs its own loss, or it never moves
codebook = nn.Parameter(torch.randn(512, 64))
z = torch.randn(10, 64, requires_grad=True)
idx = torch.cdist(z, codebook).argmin(-1)
zq = codebook[idx]
z_st = z + (zq - z).detach()
recon = (z_st ** 2).mean()                                 # stand-in for the decoder's reconstruction loss
codebook_loss = F.mse_loss(zq, z.detach())                 # moves the chosen codes towards the encoder outputs
commit = F.mse_loss(z, zq.detach())                        # keeps the encoder close to its codes
(recon + codebook_loss + 0.25 * commit).backward()
print("VQ with codebook loss: codes that receive a gradient:", int((codebook.grad.abs().sum(1) > 0).sum()),
      "(distinct codes used:", len(set(idx.tolist())), ")")

# 14.6 DDPM: the sampling step, its link to the true posterior mean, and the score
t = 500
a_t, ab_t, ab_prev, b_t = 1 - betas[t], alpha_bar[t], alpha_bar[t - 1], betas[t]
x0_, eps_ = torch.tensor(1.0), torch.tensor(0.5)
xt_ = ab_t.sqrt() * x0_ + (1 - ab_t).sqrt() * eps_
mean_ddpm = (xt_ - b_t / (1 - ab_t).sqrt() * eps_) / a_t.sqrt()           # update with the true noise as prediction
mean_post = (ab_prev.sqrt() * b_t / (1 - ab_t)) * x0_ + (a_t.sqrt() * (1 - ab_prev) / (1 - ab_t)) * xt_
print(f"t=500: beta={b_t:.5f} alpha_bar={ab_t:.4f} x_t={xt_:.3f}; DDPM mean {mean_ddpm:.4f} = posterior mean {mean_post:.4f}; "
      f"x0 recovered {(xt_ - (1 - ab_t).sqrt() * eps_) / ab_t.sqrt():.3f}; score -eps/sqrt(1-ab) = {-eps_ / (1 - ab_t).sqrt():.3f}")

# 14.6 flow matching on 1-D data with two modes at -2 and +2; sample with Euler steps from noise (t=1) to data (t=0)
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
for step in range(3000):
    x0b = torch.where(torch.rand(512, 1) < 0.5, -2.0, 2.0) + 0.3 * torch.randn(512, 1)
    epsb = torch.randn(512, 1)
    tb = torch.rand(512, 1)
    xtb = (1 - tb) * x0b + tb * epsb                          # straight path from data (t=0) to noise (t=1)
    loss_fm = ((net(torch.cat([xtb, tb], 1)) - (epsb - x0b)) ** 2).mean()   # target velocity dx/dt = eps - x0
    opt.zero_grad(); loss_fm.backward(); opt.step()
with torch.no_grad():
    for n_steps in (1, 4, 32):
        x = torch.randn(4000, 1)
        for i in range(n_steps):
            tt = torch.full_like(x, 1 - i / n_steps)
            x = x - (1 / n_steps) * net(torch.cat([x, tt], 1))   # Euler step towards t=0
        near = ((x - 2).abs() < 0.6) | ((x + 2).abs() < 0.6)
        print(f"flow matching, {n_steps:>2} Euler steps: fraction within 0.6 of a mode {near.float().mean():.2f}, "
              f"mean |x| {x.abs().mean():.2f} (data: 2.0)")

# 14.9 worked budget: vision encoder and attention compute, and time to first token
pages, side = 20, 896
pre_merge = (side // 14) ** 2                          # 4,096 patches per page before 2x2 merging
llm_tokens = pages * pre_merge // 4                    # 20,480
vit_params, vit_layers, vit_d = 0.4e9, 27, 1152        # a SigLIP-So400m-class encoder
vit_linear = 2 * vit_params * pre_merge * pages
vit_attn = 4 * vit_layers * vit_d * pre_merge ** 2 * pages      # bidirectional QK^T and AV
llm_linear = 2 * 8e9 * llm_tokens
llm_attn = 2 * 32 * 4096 * llm_tokens ** 2                       # causal: sum over positions of 4*d*t per layer
total = vit_linear + vit_attn + llm_linear + llm_attn
print(f"20-page PDF at 896 px: ViT {vit_linear:.2e} + {vit_attn:.2e}, LLM {llm_linear:.2e} + attention {llm_attn:.2e}; "
      f"total {total:.2e} FLOPs -> {total / (0.4 * 989e12):.2f} s at 40% of 989 TFLOP/s")
kv_per_token = 32 * 8 * 128 * 2 * 2
print(f"KV cache: {kv_per_token} B/token; 20,480 tokens {20480 * kv_per_token / 1e9:.2f} GB; "
      f"46,080 tokens {46080 * kv_per_token / 1e9:.2f} GB; one 1,024-token image {1024 * kv_per_token / 1e6:.0f} MB")
print("GUI: 1920x1080 at 14 px = %d patches, %d tokens after 2x2 merging, %d over 30 steps"
      % ((1920 / 14) * (1080 / 14), (1920 / 14) * (1080 / 14) / 4, 30 * (1920 / 14) * (1080 / 14) / 4))
print("paired SE, 1,000 questions, 30%% discordant, 5-point gap: %.2f points" % (100 * math.sqrt(0.30 - 0.05**2) / math.sqrt(1000)))
print("SE at 70%% on 1,000 questions: %.2f points; unpaired difference: %.2f" % (100 * math.sqrt(0.21 / 1000), 100 * math.sqrt(2 * 0.21 / 1000)))
print("one 1,024-token image: prefill %.2e FLOPs on an 8B model" % (2 * 8e9 * 1024))

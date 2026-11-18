"""Generate the book's data figures (chapters/images/*-fig-*.png) from real computations.

    uv run --python 3.12 --with numpy --with torch --with matplotlib tools/figures.py [chapter ...]

Every figure is computed here, so it can be regenerated and checked. Style: Inter, a white
surface, recessive axes, the first three validated categorical colours with direct labels and
distinct dash patterns (colour is never the only cue), a single-hue blue ramp for heatmaps.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "chapters" / "images"
FONTS = ROOT / "tools" / "fonts"

for f in ("Inter-Regular.ttf", "Inter-SemiBold.ttf"):
    font_manager.fontManager.addfont(str(FONTS / f))

INK, INK2, MUTED, GRID = "#1f1f1d", "#52514e", "#8a8882", "#e6e3dc"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # validated categorical slots 1-3
DASH = ["-", (0, (5, 2)), (0, (1.5, 1.5))]
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cfe0f6", "#6fa3e3", "#2a78d6", "#123f78"])

plt.rcParams.update({
    "font.family": "Inter", "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": "#b9b5ac", "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.titlesize": 9, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.titlecolor": INK, "legend.frameon": False, "savefig.dpi": 240,
    "figure.facecolor": "white", "axes.facecolor": "white", "lines.linewidth": 1.6,
})
W = 5.6                                            # inches: the book's text width


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", name)


def label_end(ax, x, y, text, color, dy=0, ha="left"):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", color=INK2,
                fontsize=8, va="center", ha=ha)


# ---------------------------------------------------------------------------------------------
def ch10():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # 10-fig-1: why scale by sqrt(d_k): attention collapses onto one key without it
    rng = np.random.default_rng(0)
    dks = np.array([8, 16, 32, 64, 128, 256, 512])
    def mean_max_weight(dk, scaled, trials=400, n_keys=64):
        q = rng.standard_normal((trials, dk)); k = rng.standard_normal((trials, n_keys, dk))
        s = np.einsum("td,tkd->tk", q, k) / (math.sqrt(dk) if scaled else 1.0)
        w = np.exp(s - s.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        return w.max(1).mean()
    un = [mean_max_weight(d, False) for d in dks]; sc = [mean_max_weight(d, True) for d in dks]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    ax.plot(dks, un, color=S2, ls=DASH[1], marker="o", ms=3.5)
    ax.plot(dks, sc, color=S1, ls=DASH[0], marker="o", ms=3.5)
    ax.set_xscale("log", base=2); ax.set_xticks(dks); ax.set_xticklabels(dks)
    ax.set_ylim(0, 1.05); ax.set_xlabel("head dimension d_k"); ax.set_ylabel("weight on the top key")
    ax.axhline(1 / 64, color=MUTED, lw=0.8, ls=":")
    label_end(ax, dks[-1], un[-1], "q·k (unscaled)", S2, dy=-2)
    label_end(ax, dks[-1], sc[-1], "q·k / √d_k", S1, dy=6)
    ax.text(8.3, 1 / 64 + 0.03, "uniform over 64 keys", color=MUTED, fontsize=7.5)
    ax.set_xlim(7, 1400)
    ax.set_title("Without scaling, softmax saturates as heads get wider")
    save(fig, "10-fig-1.png")

    # 10-fig-2: what heads learn: previous-token and induction heads in a trained 2-layer model
    torch.manual_seed(0)
    V, D, H, T = 32, 64, 2, 32
    def batch(n):
        x = torch.randint(0, V, (n, T)); L = torch.randint(6, 13, (n,)); s = torch.randint(0, 4, (n,))
        for i in range(n):
            a, l = int(s[i]), int(L[i]); x[i, a + l:a + 2 * l] = x[i, a:a + l]
        return x
    class AttnOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, D); self.pos = nn.Embedding(T, D)
            self.qkv = nn.ModuleList(nn.Linear(D, 3 * D, bias=False) for _ in range(2))
            self.out = nn.ModuleList(nn.Linear(D, D, bias=False) for _ in range(2))
            self.unemb = nn.Linear(D, V, bias=False); self.pat = [None, None]
        def forward(self, x):
            r = self.emb(x) + self.pos(torch.arange(T)); c = torch.ones(T, T).tril().bool()
            for l in range(2):
                q, k, v = self.qkv[l](r).view(x.shape[0], T, 3, H, D // H).permute(2, 0, 3, 1, 4)
                a = (q @ k.transpose(-1, -2) / math.sqrt(D // H)).masked_fill(~c, -1e9).softmax(-1)
                self.pat[l] = a.detach()
                r = r + self.out[l]((a @ v).transpose(1, 2).reshape(x.shape[0], T, D))
            return self.unemb(r)
    m = AttnOnly(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for _ in range(3000):
        x = batch(64); loss = F.cross_entropy(m(x)[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    first = torch.randint(0, V, (1, 10)); x = torch.cat([torch.randint(0, V, (1, 2)), first, first, torch.randint(0, V, (1, 10))], 1)
    with torch.no_grad(): m(x)
    n = 22
    pats = [m.pat[0][0, 0, :n, :n].numpy(), m.pat[1][0, 0, :n, :n].numpy()]
    # pick the layer-0 head with the strongest previous-token pattern and the layer-1 head with the strongest induction pattern
    prev_scores = [m.pat[0][0, h].diagonal(-1).mean().item() for h in range(H)]
    pats[0] = m.pat[0][0, int(np.argmax(prev_scores)), :n, :n].numpy()
    ind_scores = [m.pat[1][0, h, 12:22, 3:13].diagonal().mean().item() for h in range(H)]
    pats[1] = m.pat[1][0, int(np.argmax(ind_scores)), :n, :n].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.75))
    for ax, p, title in zip(axes, pats, ("Layer 0: previous-token head", "Layer 1: induction head")):
        ax.imshow(p, cmap=BLUES, vmin=0, vmax=1, interpolation="nearest")
        ax.grid(False); ax.set_title(title)
        ax.set_xlabel("key position (attended to)"); ax.set_ylabel("query position")
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.axvline(1.5, color=S2, lw=0.6, ls=(0, (2, 2))); ax.axhline(11.5, color=S2, lw=0.6, ls=(0, (2, 2)))
        ax.set_xticks([0, 2, 12, 21]); ax.set_yticks([0, 2, 12, 21])
    for ax in axes:
        ax.set_xticks([2, 12, 21]); ax.set_xticklabels(["2\nfirst copy\nstarts", "12\nsecond copy\nstarts", "21"], fontsize=7)
        ax.set_yticks([0, 12, 21])
    fig.tight_layout(w_pad=2.5)
    save(fig, "10-fig-2.png")

    # 10-fig-3: RoPE wavelengths per dimension pair, and how context extension changes them
    d = 128; i = np.arange(d // 2)
    base = 10000.0; s = 8
    wl = 2 * np.pi * base ** (2 * i / d)
    wl_pi = wl * s
    wl_ntk = 2 * np.pi * (base * s ** (d / (d - 2))) ** (2 * i / d)
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for y, c, ls, lab, xi, dy in ((wl, S1, DASH[0], "RoPE, base 10,000", 44, -10), (wl_pi, S2, DASH[1], "position interpolation ×8", 1, 12),
                                  (wl_ntk, S3, DASH[2], "NTK-aware ×8", 31, -14)):
        ax.plot(i, y, color=c, ls=ls)
        if xi == 1:
            ax.text(1, 2500, lab, color=INK2, fontsize=8)
        else:
            ax.annotate(lab, (i[xi], y[xi]), xytext=(4, dy), textcoords="offset points", color=INK2, fontsize=8)
    ax.set_yscale("log"); ax.set_xlim(0, 63)
    for ctx, txt in ((4096, "trained context 4k"), (32768, "target 32k")):
        ax.axhline(ctx, color=MUTED, lw=0.7, ls=":")
        ax.text(62, ctx * 1.15, txt, color=MUTED, fontsize=7.5, ha="right")
    ax.set_xlabel("dimension pair i (of 64)"); ax.set_ylabel("wavelength (tokens)")
    ax.set_title("Each RoPE pair rotates at its own frequency")
    save(fig, "10-fig-3.png")

    # 10-fig-4: KV-cache memory per sequence vs context, for an 8B-shaped model in bf16
    ctx = np.linspace(0, 131072, 200)
    def gb(kv): return 2 * 32 * kv * 128 * ctx * 2 / 1e9
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for kv, c, ls, lab in ((32, S2, DASH[1], "MHA, 32 KV heads"), (8, S1, DASH[0], "GQA, 8 KV heads"), (1, S3, DASH[2], "MQA, 1 KV head")):
        y = gb(kv); ax.plot(ctx / 1024, y, color=c, ls=ls)
        label_end(ax, 128, y[-1], f"{lab}: {y[-1]:.0f} GB", c, dy=-6 if kv == 8 else 0)
    ax.hlines(16, 0, 128, color=MUTED, lw=0.8, ls=":"); ax.text(2, 18, "the model's own weights (8B in bf16): 16 GB", color=MUTED, fontsize=7.5)
    ax.set_xlim(0, 175); ax.set_xticks([0, 32, 64, 96, 128])
    ax.set_xlabel("context length (thousands of tokens)"); ax.set_ylabel("KV cache per sequence (GB)")
    ax.set_title("At long context the cache, not the weights, fills the GPU")
    save(fig, "10-fig-4.png")

    # 10-fig-5: where the parameters of an 8B-shaped decoder live
    vocab, dm, L, kvh, hd, ff = 128256, 4096, 32, 8, 128, 14336
    parts = [("Input embedding", vocab * dm), ("Attention (Q, K, V, O)", L * (dm * dm * 2 + dm * kvh * hd * 2)),
             ("Feed-forward (SwiGLU)", L * 3 * dm * ff), ("Output head (untied)", vocab * dm)]
    total = sum(v for _, v in parts) + L * 2 * dm + dm
    fig, ax = plt.subplots(figsize=(W, 1.75))
    names = [p[0] for p in parts][::-1]; vals = [p[1] / 1e9 for p in parts][::-1]
    ax.barh(names, vals, color=S1, height=0.55)
    for y, v in enumerate(vals):
        ax.text(v + 0.06, y, f"{v:.2f}B ({v * 1e9 / total:.0%})", va="center", color=INK2, fontsize=8)
    ax.set_xlim(0, 7); ax.grid(axis="y", visible=False); ax.set_xlabel("parameters (billions)")
    ax.set_title(f"An 8B-shaped decoder: {total / 1e9:.2f}B parameters in total")
    save(fig, "10-fig-5.png")


# ---------------------------------------------------------------------------------------------
def ch18():
    import torch
    import torch.nn.functional as F

    # 18-fig-1: the PPO clipped objective as a function of the probability ratio
    r = np.linspace(0.3, 1.8, 400); eps = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.2), sharey=False)
    for ax, A, title in ((axes[0], 1.0, "Advantage > 0 (a good token)"), (axes[1], -1.0, "Advantage < 0 (a bad token)")):
        unclipped = r * A; clipped = np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A)
        ax.plot(r, unclipped, color=MUTED, ls=DASH[1], lw=1.1)
        ax.plot(r, clipped, color=S1)
        ax.axvline(1, color=GRID, lw=0.8); ax.axvspan(1 - eps, 1 + eps, color="#eef3fb", lw=0)
        ax.set_xlabel("ratio π_new / π_old"); ax.set_title(title)
    axes[0].set_ylabel("objective")
    axes[0].annotate("gain capped at 1 + ε", (1.5, 1.2), xytext=(0, -13), textcoords="offset points", color=INK2, fontsize=7.5, ha="center")
    axes[1].text(0.3, -1.35, "penalty capped at 1 − ε", color=INK2, fontsize=7.5, ha="left")
    axes[0].text(1.08, 0.45, "solid: clipped objective\ndashed: unclipped ρ·A", color=INK2, fontsize=7, va="bottom")
    fig.tight_layout(w_pad=2)
    save(fig, "18-fig-1.png")

    # 18-fig-2: reward over-optimisation with best-of-n (same simulation as code/ch18_check.py)
    rng = np.random.default_rng(0)
    n_samples = 400_000
    quality = rng.normal(size=n_samples); length = rng.normal(size=n_samples)
    gold = quality - 0.5 * length ** 2; proxy = quality + 1.0 * length
    ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]; kl, pv, gv = [], [], []
    for n in ns:
        k = n_samples // n
        idx = proxy[: k * n].reshape(k, n).argmax(1) + np.arange(k) * n
        kl.append(math.log(n) - (n - 1) / n); pv.append(proxy[idx].mean()); gv.append(gold[idx].mean())
    fig, ax = plt.subplots(figsize=(W, 2.5))
    ax.plot(kl, pv, color=S2, ls=DASH[1], marker="o", ms=3)
    ax.plot(kl, gv, color=S1, ls=DASH[0], marker="o", ms=3)
    label_end(ax, kl[-1], pv[-1], "proxy reward (what we optimise)", S2)
    label_end(ax, kl[-1], gv[-1], "gold reward (what people want)", S1)
    best = int(np.argmax(gv)); ax.annotate(f"gold peaks at best-of-{ns[best]}", (kl[best], gv[best]), xytext=(0, -16),
                                           textcoords="offset points", ha="center", color=INK2, fontsize=7.5)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xlim(-0.1, 9.5); ax.set_xlabel("KL from the starting policy (nats), best-of-n"); ax.set_ylabel("mean reward")
    ax.set_title("Optimising a proxy reward: gold reward rises, then falls")
    save(fig, "18-fig-2.png")

    # 18-fig-3: DPO training on a bandit converges to the closed-form optimum
    rr = torch.tensor([1.0, 0.5, 0.0, -1.0]); beta = 0.5; pi_ref = torch.tensor([0.1, 0.2, 0.3, 0.4])
    target = pi_ref * torch.exp(rr / beta); target /= target.sum()
    logits = pi_ref.log().clone().requires_grad_(True); opt = torch.optim.Adam([logits], lr=0.05)
    pairs = [(i, j) for i in range(4) for j in range(4) if i != j]; hist = []
    for step in range(1500):
        logp = F.log_softmax(logits, 0)
        loss = sum(-torch.sigmoid(rr[i] - rr[j]) * F.logsigmoid(beta * ((logp[i] - pi_ref[i].log()) - (logp[j] - pi_ref[j].log())))
                   for i, j in pairs) / len(pairs)
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(F.softmax(logits, 0).detach().numpy().copy())
    hist = np.array(hist)
    fig, ax = plt.subplots(figsize=(W, 2.4))
    colors = [S1, S2, S3, MUTED]
    steps = 400
    for k in range(4):
        ax.plot(hist[:steps, k], color=colors[k], ls=DASH[k % 3] if k < 3 else (0, (3, 1, 1, 1)))
        ax.hlines(target[k].item(), 0, steps, color=colors[k], lw=0.7, ls=":")
        label_end(ax, steps, hist[-1, k], f"response {k + 1}, reward {rr[k]:+.1f}: {hist[-1, k]:.3f} (optimum {target[k]:.3f})", colors[k])
    ax.set_xlim(0, 820); ax.set_xticks([0, 100, 200, 300, 400]); ax.set_xlabel("DPO training step"); ax.set_ylabel("policy probability")
    ax.set_title("DPO converges to π_ref · exp(r/β), the dotted lines")
    save(fig, "18-fig-3.png")


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["10", "18"]
    for ch in wanted:
        globals()[f"ch{ch}"]()

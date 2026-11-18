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

# ---------------------------------------------------------------------------------------------
def ch01():
    rng = np.random.default_rng(0)
    # 01-fig-1: cosine similarity between random vectors concentrates around 0 as dimension grows
    fig, ax = plt.subplots(figsize=(W, 2.3))
    bins = np.linspace(-1, 1, 81)
    for d, c, ls in ((3, S2, DASH[1]), (64, S3, DASH[2]), (768, S1, DASH[0])):
        a = rng.standard_normal((20000, d)); b = rng.standard_normal((20000, d))
        cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
        h, e = np.histogram(cos, bins=bins, density=True)
        ax.plot((e[:-1] + e[1:]) / 2, h, color=c, ls=ls)
        pos = {3: (0.55, 1.1), 64: (0.2, 2.6), 768: (0.07, 9.0)}[d]
        ax.text(*pos, f"d = {d}: sd {cos.std():.3f}", color=INK2, fontsize=8)
        print(f"cosine of random vectors, d={d}: sd {cos.std():.3f} (1/sqrt(d) = {1/np.sqrt(d):.3f})")
    ax.set_xlabel("cosine similarity of two random vectors"); ax.set_ylabel("density")
    ax.set_title("In high dimensions, random vectors are almost orthogonal")
    save(fig, "01-fig-1.png")

    # 01-fig-2: precision of a detector as a function of prevalence
    prev = np.logspace(-4, -0.3, 200)
    fig, ax = plt.subplots(figsize=(W, 2.3))
    for fpr, c, ls in ((0.05, S2, DASH[1]), (0.01, S1, DASH[0])):
        prec = 0.95 * prev / (0.95 * prev + fpr * (1 - prev))
        ax.plot(prev * 100, prec, color=c, ls=ls)
        i = 120; ax.annotate(f"recall 95%, false-positive rate {fpr:.0%}", (prev[i] * 100, prec[i]),
                             xytext=(6, -10 if fpr == 0.05 else 6), textcoords="offset points", color=INK2, fontsize=8)
    for p in (0.001, 0.01, 0.1):
        print(f"precision at prevalence {p:.1%}: fpr5% {0.95*p/(0.95*p+0.05*(1-p)):.3f}, fpr1% {0.95*p/(0.95*p+0.01*(1-p)):.3f}")
    ax.set_xscale("log"); ax.set_xlabel("prevalence of positives (%)"); ax.set_ylabel("precision")
    ax.set_xticks([0.01, 0.1, 1, 10]); ax.set_xticklabels(["0.01", "0.1", "1", "10"])
    ax.set_title("The same detector, different base rates: precision collapses for rare events")
    save(fig, "01-fig-2.png")


def ch02():
    from scipy import stats
    rng = np.random.default_rng(0)
    # 02-fig-1: distribution of p-values with no effect vs with a real effect
    def pvals(effect, n=500, trials=6000):
        a = rng.normal(0, 1, (trials, n)); b = rng.normal(effect, 1, (trials, n))
        return stats.ttest_ind(a, b, axis=1).pvalue
    p0, p1 = pvals(0.0), pvals(0.15)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.1), sharey=True)
    for ax, p, title in ((axes[0], p0, "No effect: p-values are uniform"), (axes[1], p1, "Real effect: p-values pile up near 0")):
        ax.hist(p, bins=np.linspace(0, 1, 21), color=S1, edgecolor="white", linewidth=1)
        ax.axvline(0.05, color=S2, lw=1, ls=DASH[1]); ax.set_title(title); ax.set_xlabel("p-value")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("experiments")
    print(f"share p<0.05: null {np.mean(p0 < 0.05):.3f}, effect 0.15 sd with n=500/arm {np.mean(p1 < 0.05):.3f}")
    fig.tight_layout(w_pad=1.5); save(fig, "02-fig-1.png")

    # 02-fig-2: power against sample size for a 5% baseline conversion
    n = np.logspace(3, 6.3, 300); za = stats.norm.ppf(0.975)
    fig, ax = plt.subplots(figsize=(W, 2.3))
    for mde, c, ls in ((0.10, S1, DASH[0]), (0.05, S2, DASH[1]), (0.02, S3, DASH[2])):
        p1_, p2_ = 0.05, 0.05 * (1 + mde)
        se = np.sqrt((p1_ * (1 - p1_) + p2_ * (1 - p2_)) / n)
        power = stats.norm.cdf((p2_ - p1_) / se - za)
        ax.plot(n, power, color=c, ls=ls)
        k = np.searchsorted(power, 0.8)
        ax.annotate(f"+{mde:.0%} lift: {n[k]/1000:,.0f}k", (n[k], 0.8), xytext=(-5, 5), ha="right",
                    textcoords="offset points", color=INK2, fontsize=7.5)
    ax.axhline(0.8, color=MUTED, lw=0.7, ls=":")
    ax.set_xscale("log"); ax.set_xlim(1000, 2.5e6); ax.set_xlabel("users per arm (labels: users needed for 80% power)"); ax.set_ylabel("power")
    ax.set_title("Halving the effect you want to detect quadruples the users you need")
    save(fig, "02-fig-2.png")

    # 02-fig-3: peeking inflates false positives
    looks = [1, 2, 5, 10, 20, 50]
    rates = []
    for k in looks:
        hits = 0; trials = 3000; n_total = 10000
        for _ in range(trials):
            a = rng.normal(0, 1, n_total); b = rng.normal(0, 1, n_total)
            checkpoints = np.linspace(n_total / k, n_total, k).astype(int)
            ca, cb = np.cumsum(a), np.cumsum(b); ca2, cb2 = np.cumsum(a ** 2), np.cumsum(b ** 2)
            for m in checkpoints:
                ma, mb = ca[m - 1] / m, cb[m - 1] / m
                va = ca2[m - 1] / m - ma ** 2; vb = cb2[m - 1] / m - mb ** 2
                z = (mb - ma) / np.sqrt((va + vb) / m)
                if abs(z) > 1.96: hits += 1; break
        rates.append(hits / trials)
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.plot(looks, rates, color=S2, marker="o", ms=3.5)
    for x, y in zip(looks, rates):
        ax.annotate(f"{y:.0%}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", color=INK2, fontsize=7.5)
    ax.axhline(0.05, color=MUTED, lw=0.7, ls=":"); ax.text(30, 0.02, "nominal 5%", color=MUTED, fontsize=7.5)
    ax.set_xscale("log"); ax.set_xticks(looks); ax.set_xticklabels(looks); ax.set_ylim(0, max(rates) + 0.08)
    ax.set_xlabel("number of looks, stopping at the first p < 0.05"); ax.set_ylabel("false positive rate")
    ax.set_title("A/A tests (no real effect): every extra look adds false wins")
    print("peeking false positive rates:", dict(zip(looks, [round(r, 3) for r in rates])))
    save(fig, "02-fig-3.png")


def ch03():
    rng = np.random.default_rng(0)
    # 03-fig-1: bias^2, variance and test error vs polynomial degree (repeated training sets)
    f_true = lambda x: np.sin(2.5 * x)
    x_test = np.linspace(-1, 1, 200); degrees = list(range(1, 10)); noise = 0.3
    b2, var, test = [], [], []
    for deg in degrees:
        preds = []
        for _ in range(300):
            x = rng.uniform(-1, 1, 25); y = f_true(x) + rng.normal(0, noise, 25)
            coef = np.polyfit(x, y, deg); preds.append(np.polyval(coef, x_test))
        preds = np.clip(np.array(preds), -5, 5)
        mean = preds.mean(0)
        b2.append(np.mean((mean - f_true(x_test)) ** 2)); var.append(np.mean(preds.var(0)))
        test.append(b2[-1] + var[-1] + noise ** 2)
    fig, ax = plt.subplots(figsize=(W, 2.4))
    for y, c, ls, lab in ((b2, S2, DASH[1], "bias²"), (var, S3, DASH[2], "variance"), (test, S1, DASH[0], "expected test error")):
        ax.plot(degrees, y, color=c, ls=ls, marker="o", ms=3)
        k = 6 if lab != "bias²" else 8
        ax.annotate(lab, (degrees[k], min(y[k], 0.85)), xytext=(-6, 2), textcoords="offset points", ha="right", color=INK2, fontsize=8)
    best = int(np.argmin(test)); ax.annotate(f"best: degree {degrees[best]}", (degrees[best], test[best]), xytext=(0, 8),
                                             textcoords="offset points", ha="center", color=INK2, fontsize=7.5)
    ax.axhline(noise ** 2, color=MUTED, lw=0.7, ls=":"); ax.text(1, noise ** 2 + 0.02, "irreducible noise", color=MUTED, fontsize=7.5)
    ax.set_ylim(0, 0.9); ax.set_xlim(0.5, 9.5); ax.set_xticks(degrees)
    ax.set_xlabel("polynomial degree (model flexibility)"); ax.set_ylabel("mean squared error")
    ax.set_title("Flexibility trades bias for variance (25 training points, 300 resamples)")
    print("bias-variance:", [(d, round(b, 3), round(v, 3)) for d, b, v in zip(degrees, b2, var)][:8], "best degree", degrees[best])
    save(fig, "03-fig-1.png")

    # 03-fig-2: gradient descent vs momentum on an elongated quadratic
    A = np.array([1.0, 25.0])                         # curvatures along x and y
    def run(lr, beta, steps=60):
        w = np.array([-4.5, 1.2]); v = np.zeros(2); path = [w.copy()]
        for _ in range(steps):
            g = A * w; v = beta * v + g; w = w - lr * v; path.append(w.copy())
        return np.array(path)
    gd, mom = run(0.075, 0.0), run(0.03, 0.8)
    xx, yy = np.meshgrid(np.linspace(-5, 1, 200), np.linspace(-1.6, 1.6, 200))
    fig, ax = plt.subplots(figsize=(W, 2.4))
    ax.contour(xx, yy, 0.5 * (A[0] * xx ** 2 + A[1] * yy ** 2), levels=[0.5, 2, 5, 10, 20], colors=GRID, linewidths=0.8)
    ax.plot(gd[:30, 0], gd[:30, 1], color=S2, ls=DASH[1], marker="o", ms=2.5, lw=1.1)
    ax.plot(mom[:30, 0], mom[:30, 1], color=S1, marker="o", ms=2.5, lw=1.1)
    ax.text(-1.6, 1.25, "dashed: gradient descent, lr 0.075\nsolid: momentum 0.8, lr 0.03", color=INK2, fontsize=7.5, va="top")
    ax.grid(False); ax.set_xlabel("w₁ (low curvature)"); ax.set_ylabel("w₂ (high curvature)")
    ax.set_title("A narrow valley: momentum averages out the oscillation")
    loss = lambda p: 0.5 * (A[0] * p[:, 0] ** 2 + A[1] * p[:, 1] ** 2)
    print(f"loss after 30 steps: GD {loss(gd)[30]:.4f}, momentum {loss(mom)[30]:.4f}")
    save(fig, "03-fig-2.png")


def ch04():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    # 04-fig-1: ROC vs PR for two models under heavy imbalance
    X, y = make_classification(n_samples=60000, n_features=20, n_informative=6, weights=[0.995], flip_y=0.001, class_sep=0.9, random_state=1)
    Xtr, Xte, ytr, yte = X[:30000], X[30000:], y[:30000], y[30000:]
    m1 = LogisticRegression(max_iter=2000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    m2 = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5))
    for p, c, ls, name in ((m1, S2, DASH[1], "logistic regression"), (m2, S1, DASH[0], "random forest")):
        fpr, tpr, _ = roc_curve(yte, p); pr, rc, _ = precision_recall_curve(yte, p)
        axes[0].plot(fpr, tpr, color=c, ls=ls, label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
        axes[1].plot(rc, pr, color=c, ls=ls, label=f"{name} (AP {average_precision_score(yte, p):.3f})")
        print(f"{name}: ROC-AUC {roc_auc_score(yte, p):.3f}, AP {average_precision_score(yte, p):.3f}, positives {yte.mean():.4f}")
    axes[0].plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=":")
    axes[1].axhline(yte.mean(), color=MUTED, lw=0.7, ls=":")
    axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("recall"); axes[0].set_title("ROC: both look strong")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title("PR: the gap is visible")
    axes[0].legend(fontsize=7, loc="lower right"); axes[1].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(w_pad=2); save(fig, "04-fig-1.png")

    # 04-fig-2: reliability diagram before and after Platt scaling
    rng = np.random.default_rng(0)
    true_p = rng.beta(1.2, 5, 40000); yy = (rng.random(40000) < true_p).astype(int)
    raw = true_p ** 0.45                                # an over-confident, monotone distortion
    cal = LogisticRegression().fit(np.log(raw[:20000] / (1 - raw[:20000] + 1e-9)).reshape(-1, 1), yy[:20000])
    fixed = cal.predict_proba(np.log(raw[20000:] / (1 - raw[20000:] + 1e-9)).reshape(-1, 1))[:, 1]
    def curve(p, y):
        b = np.minimum((p * 10).astype(int), 9); xs, ys, ws = [], [], []
        for k in range(10):
            m = b == k
            if m.sum() > 50: xs.append(p[m].mean()); ys.append(y[m].mean()); ws.append(m.mean())
        ece = sum(w * abs(a - c) for a, c, w in zip(xs, ys, ws))
        return xs, ys, ece
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.6))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=":")
    for p, y_, c, ls, name in ((raw[20000:], yy[20000:], S2, DASH[1], "raw scores"), (fixed, yy[20000:], S1, DASH[0], "after Platt scaling")):
        xs, ys, ece = curve(p, y_); ax.plot(xs, ys, color=c, ls=ls, marker="o", ms=3)
        ax.text(0.03 if name.startswith("after") else 0.42, 0.85 if name.startswith("after") else 0.12,
                f"{'solid' if name.startswith('after') else 'dashed'}: {name}, ECE {ece:.3f}", color=INK2, fontsize=7.5)
        print(f"{name}: ECE {ece:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Reliability diagram")
    save(fig, "04-fig-2.png")

# ---------------------------------------------------------------------------------------------
def ch05():
    import torch, torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0); torch.manual_seed(0)
    # spiral data, 3 classes
    n = 200; X, y = [], []
    for c in range(3):
        r = np.linspace(0.05, 1, n); t = np.linspace(c * 4, (c + 1) * 4, n) + rng.normal(0, 0.2, n)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)]); y += [c] * n
    X = np.vstack(X); y = np.array(y)
    lr = LogisticRegression(max_iter=2000).fit(X, y)
    mlp = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 3))
    opt = torch.optim.Adam(mlp.parameters(), lr=0.02); Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y)
    for _ in range(1500):
        loss = nn.functional.cross_entropy(mlp(Xt), yt); opt.zero_grad(); loss.backward(); opt.step()
    xx, yy = np.meshgrid(np.linspace(-1.1, 1.1, 250), np.linspace(-1.1, 1.1, 250)); grid = np.c_[xx.ravel(), yy.ravel()]
    cmap = matplotlib.colors.ListedColormap(["#d6e6f8", "#fbdccd", "#cdeee1"])
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7))
    for ax, pred, title in ((axes[0], lr.predict(grid), f"Logistic regression: {lr.score(X, y):.0%} accuracy"),
                            (axes[1], mlp(torch.tensor(grid, dtype=torch.float32)).argmax(1).numpy(),
                             f"MLP, 64 hidden units: {(mlp(Xt).argmax(1).numpy() == y).mean():.0%} accuracy")):
        ax.contourf(xx, yy, pred.reshape(xx.shape), levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap)
        for c, col, mk in ((0, S1, "o"), (1, S2, "s"), (2, S3, "^")):
            ax.scatter(X[y == c, 0], X[y == c, 1], s=5, color=col, marker=mk, linewidths=0)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_aspect("equal")
        for sp in ax.spines.values(): sp.set_visible(False)
    fig.tight_layout(w_pad=1.5); save(fig, "05-fig-1.png")

    # 05-fig-2: activation scale through 20 ReLU layers under three initialisations
    xin = rng.standard_normal((512, 256)); fig, ax = plt.subplots(figsize=(W, 2.3))
    for name, std_fn, c, ls in (("std 0.01", lambda d: 0.01, S2, DASH[1]), ("std 1.0", lambda d: 1.0, S3, DASH[2]),
                                ("He: std √(2/fan_in)", lambda d: np.sqrt(2 / d), S1, DASH[0])):
        h = xin; stds = [h.std()]
        for _ in range(20):
            h = np.maximum(0, h @ rng.normal(0, std_fn(256), (256, 256))); stds.append(h.std())
        ax.plot(range(21), stds, color=c, ls=ls, marker="o", ms=2.5)
        label_end(ax, 20, stds[-1], f"{name}: {stds[-1]:.1e}", c)
        print(f"init {name}: std after 20 layers {stds[-1]:.2e}")
    ax.set_yscale("log"); ax.set_xlim(0, 30); ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("layer"); ax.set_ylabel("std of activations"); ax.set_title("Only the right scale keeps the signal alive through depth")
    save(fig, "05-fig-2.png")


def ch06():
    import torch, torch.nn as nn
    torch.manual_seed(0)
    # 06-fig-1: learning-rate range test on a small MLP
    X = torch.randn(4096, 20); w = torch.randn(20); y = (X @ w + 0.5 * torch.randn(4096) > 0).long()
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2)); opt = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    lrs = np.logspace(-5, 1, 120); losses = []; smooth = None
    for i, lr_ in enumerate(lrs):
        for g in opt.param_groups: g["lr"] = lr_
        idx = torch.randint(0, 4096, (64,))
        loss = nn.functional.cross_entropy(model(X[idx]), y[idx]); opt.zero_grad(); loss.backward(); opt.step()
        smooth = loss.item() if smooth is None else 0.9 * smooth + 0.1 * loss.item(); losses.append(smooth)
        if smooth > 4 * min(losses): break
    lrs = lrs[:len(losses)]; best = lrs[int(np.argmin(losses))]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.plot(lrs, losses, color=S1); ax.set_xscale("log")
    ax.axvline(best / 10, color=S2, lw=0.8, ls=DASH[1])
    ax.annotate(f"loss lowest near lr = {best:.2g}\npick about 10× lower: {best/10:.2g}", (best / 10, min(losses)),
                xytext=(-8, 12), textcoords="offset points", ha="right", color=INK2, fontsize=7.5)
    ax.set_xlabel("learning rate (increased every step)"); ax.set_ylabel("smoothed training loss")
    ax.set_title("A learning-rate range test in one pass")
    print(f"LR range test: min loss at lr {best:.3g}")
    save(fig, "06-fig-1.png")


def ch07():
    import torch, torch.nn as nn
    # 07-fig-1: cosine vs WSD schedules
    total = 1000; steps = np.arange(total)
    def cosine(s, peak=3e-4, warm=100, final=0.1):
        return np.where(s < warm, peak * s / warm, peak * (final + (1 - final) * 0.5 * (1 + np.cos(np.pi * (s - warm) / (total - warm)))))
    def wsd(s, peak=3e-4, warm=100, frac=0.2):
        ds = int(total * (1 - frac))
        return np.where(s < warm, peak * s / warm, np.where(s < ds, peak, peak * (1 - (s - ds) / (total - ds))))
    fig, ax = plt.subplots(figsize=(W, 2.1))
    ax.plot(steps, cosine(steps) * 1e4, color=S2, ls=DASH[1]); ax.plot(steps, wsd(steps) * 1e4, color=S1)
    br = 500; ds = 800
    ax.plot([br, br + 100], [3, 0], color=S3, ls=DASH[2])
    ax.annotate("cosine", (300, cosine(np.array([300]))[0] * 1e4), xytext=(-4, -12), textcoords="offset points", color=INK2, fontsize=8, ha="right")
    ax.annotate("warmup–stable–decay", (650, 3.0), xytext=(0, 5), textcoords="offset points", color=INK2, fontsize=8, ha="center")
    ax.annotate("a branch decayed\nfrom step 500", (550, 1.5), xytext=(10, 0), textcoords="offset points", color=INK2, fontsize=7.5)
    ax.set_xlabel("training step"); ax.set_ylabel("learning rate (×10⁻⁴)"); ax.set_ylim(0, 3.6)
    ax.set_title("Schedules: WSD stays at the peak and can be decayed from any checkpoint")
    save(fig, "07-fig-1.png")

    # 07-fig-2: one Adam step's effect on the output vs width, standard vs muP (same set-up as code/ch07_check.py)
    def output_change(width, mup):
        torch.manual_seed(1); x = torch.randn(256, 64)
        l1 = nn.Linear(64, width, bias=False); l2 = nn.Linear(width, width, bias=False); l3 = nn.Linear(width, 1, bias=False)
        nn.init.normal_(l1.weight, std=1 / np.sqrt(64)); nn.init.normal_(l2.weight, std=1 / np.sqrt(width))
        nn.init.normal_(l3.weight, std=(1 / width) if mup else (1 / np.sqrt(width)))
        net = nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU(), l3)
        opt = torch.optim.Adam([{"params": l1.parameters(), "lr": 1e-3},
                                {"params": list(l2.parameters()) + list(l3.parameters()), "lr": 1e-3 * (64 / width if mup else 1)}])
        out0 = net(x).detach(); loss = (net(x) - 1).pow(2).mean(); loss.backward(); opt.step()
        return (net(x).detach() - out0).abs().mean().item()
    widths = [64, 128, 256, 512, 1024, 2048]
    sp = [output_change(w_, False) for w_ in widths]; mp = [output_change(w_, True) for w_ in widths]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.plot(widths, sp, color=S2, ls=DASH[1], marker="o", ms=3); ax.plot(widths, mp, color=S1, marker="o", ms=3)
    label_end(ax, widths[-1], sp[-1], "standard parametrisation", S2); label_end(ax, widths[-1], mp[-1], "µP", S1)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(widths); ax.set_xticklabels(widths); ax.set_xlim(50, 9000)
    ax.set_xlabel("width"); ax.set_ylabel("mean change in output")
    ax.set_title("Same learning rate, one Adam step: µP keeps the update size constant")
    print("muP:", [round(v, 3) for v in mp], "SP:", [round(v, 3) for v in sp])
    save(fig, "07-fig-2.png")


def ch08():
    import torch, torch.nn as nn
    torch.manual_seed(0)
    # 08-fig-1: how much the last state depends on earlier inputs: vanilla RNN vs LSTM (random init, forget bias 1)
    T, d = 60, 32
    def grad_profile(cell):
        x = torch.randn(1, T, d, requires_grad=True)
        out, _ = cell(x); out[0, -1].sum().backward()
        return x.grad[0].norm(dim=1).numpy()[::-1]          # index k = k steps before the end
    rnn = nn.RNN(d, d, batch_first=True); lstm = nn.LSTM(d, d, batch_first=True)
    with torch.no_grad():
        lstm.bias_ih_l0[d:2 * d] = 1.0; lstm.bias_hh_l0[d:2 * d] = 0.0
    g_rnn = np.mean([grad_profile(rnn) for _ in range(20)], 0); g_lstm = np.mean([grad_profile(lstm) for _ in range(20)], 0)
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.plot(range(T), g_rnn / g_rnn[0], color=S2, ls=DASH[1]); ax.plot(range(T), g_lstm / g_lstm[0], color=S1)
    label_end(ax, 40, (g_rnn / g_rnn[0])[40], "vanilla RNN (tanh)", S2); label_end(ax, 40, (g_lstm / g_lstm[0])[40], "LSTM, forget-gate bias 1", S1, dy=9)
    ax.set_yscale("log"); ax.set_xlabel("steps back from the last position"); ax.set_ylabel("gradient size (relative)")
    ax.set_title("Gradients from far-back inputs vanish much faster in a plain RNN")
    print(f"gradient after 20 steps: RNN {(g_rnn/g_rnn[0])[20]:.1e}, LSTM {(g_lstm/g_lstm[0])[20]:.1e}")
    save(fig, "08-fig-1.png")

# ---------------------------------------------------------------------------------------------
def ch09():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted((ROOT / "chapters").glob("*.md")))
    lines = [l for l in text.split("\n") if l.strip() and not l.startswith("```")]
    train, test = lines[: int(len(lines) * 0.9)], lines[int(len(lines) * 0.9):]
    sizes = [300, 1000, 2000, 4000, 8000, 16000, 32000]; cpt = []
    for v in sizes:
        tok = Tokenizer(models.BPE()); tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.train_from_iterator(train, trainers.BpeTrainer(vocab_size=v, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
        n_tok = sum(len(tok.encode(l).ids) for l in test); n_chr = sum(len(l) for l in test)
        cpt.append(n_chr / n_tok)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.2))
    axes[0].plot(sizes, cpt, color=S1, marker="o", ms=3); axes[0].set_xscale("log")
    for x, y in zip(sizes, cpt):
        if x in (300, 4000, 32000): axes[0].annotate(f"{y:.2f}", (x, y), xytext=(0, 6), textcoords="offset points", ha="center", color=INK2, fontsize=7.5)
    axes[0].set_ylim(1, 4.6); axes[0].set_xlabel("vocabulary size"); axes[0].set_ylabel("characters per token"); axes[0].set_title("Compression: diminishing returns")
    emb = [2 * v * 4096 / 1e6 for v in sizes]
    axes[1].plot(sizes, emb, color=S2, ls=DASH[1], marker="o", ms=3); axes[1].set_xscale("log")
    axes[1].set_xlabel("vocabulary size"); axes[1].set_ylabel("embedding + output params (M)"); axes[1].set_title("Cost: linear in vocabulary (d = 4,096)")
    fig.tight_layout(w_pad=2); save(fig, "09-fig-1.png")
    print("chars/token by vocab:", dict(zip(sizes, [round(c, 2) for c in cpt])))


def ch11():
    rng = np.random.default_rng(0)
    # 11-fig-1: expert load before and after auxiliary-loss-free balancing (same simulation as code/ch11_check.py idea)
    n = 8; scores = rng.standard_normal((4096, n)) + np.array([1.5, 1.0, 0.5, 0, 0, 0, -0.5, -1.0])
    bias = np.zeros(n); loads = []
    for step in range(200):
        top = np.argsort(-(scores + bias), axis=1)[:, :2]
        load = np.bincount(top.ravel(), minlength=n) / top.size
        if step == 0: loads.append(load)
        bias -= 0.01 * np.sign(load - 1 / n)
    loads.append(load)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.1), sharey=True)
    for ax, l, t in ((axes[0], loads[0], "Before: router prefers a few experts"), (axes[1], loads[1], "After 200 bias updates")):
        ax.bar(range(1, n + 1), l, color=S1, width=0.6); ax.axhline(1 / n, color=S2, lw=0.9, ls=DASH[1])
        ax.set_title(t); ax.set_xlabel("expert"); ax.set_xticks(range(1, n + 1)); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("share of routed tokens"); axes[1].text(8.4, 1 / n, "uniform", color=INK2, fontsize=7.5, va="center")
    fig.tight_layout(w_pad=1.5); save(fig, "11-fig-1.png")

    # 11-fig-2: memory per sequence vs context: full attention (GQA) vs recurrent state vs a 1:3 hybrid
    T = np.linspace(0, 1_048_576, 300); L, kvh, hd = 32, 8, 128
    att = 2 * 2 * L * kvh * hd * T / 1e9
    state = np.full_like(T, 2 * L * 32 * 128 * 128 / 1e9)
    hybrid = att / 4 + state * 3 / 4
    fig, ax = plt.subplots(figsize=(W, 2.3))
    for y, c, ls, lab in ((att, S2, DASH[1], "all attention layers (GQA-8)"), (hybrid, S1, DASH[0], "hybrid: 1 attention layer in 4"),
                          (state, S3, DASH[2], "all recurrent layers (fixed state)")):
        ax.plot(T / 1024, y, color=c, ls=ls); label_end(ax, 1024, y[-1], f"{lab}: {y[-1]:.0f} GB" if y[-1] > 1 else f"{lab}: {y[-1]:.2f} GB", c)
    ax.set_xlim(0, 1700); ax.set_xticks([0, 256, 512, 768, 1024])
    ax.set_xlabel("context length (thousands of tokens)"); ax.set_ylabel("memory per sequence (GB)")
    ax.set_title("Why hybrids: the cache shrinks with the share of attention layers")
    save(fig, "11-fig-2.png")


def ch12():
    models_ = [("BERT-base encoder\n110M", 110e6), ("BERT-large encoder\n340M", 340e6), ("8B LLM", 8e9), ("70B LLM", 70e9)]
    tokens = 64
    flops = [2 * p * tokens for _, p in models_]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    names = [m[0] for m in models_][::-1]; vals = [f_ / 1e9 for f_ in flops][::-1]
    ax.barh(names, vals, color=S1, height=0.55); ax.set_xscale("log")
    for y, v in enumerate(vals):
        ax.text(v * 1.15, y, f"{v:,.0f} GFLOPs" if v >= 10 else f"{v:.1f} GFLOPs", va="center", color=INK2, fontsize=8)
    ax.set_xlim(5, 3e5); ax.grid(axis="y", visible=False); ax.set_xlabel("compute per 64-token classification (log scale)")
    ax.set_title("Classifying one short text: about 640× more compute for a 70B LLM than BERT-base")
    save(fig, "12-fig-1.png")


def ch13():
    E, A, B, al, be = 1.69, 406.4, 410.7, 0.34, 0.28
    Lf = lambda N, D: E + A / N ** al + B / D ** be
    # 13-fig-1: IsoFLOP curves
    fig, ax = plt.subplots(figsize=(W, 2.5)); mins = []
    for C, c, ls in ((1e19, S3, DASH[2]), (1e21, S2, DASH[1]), (1e23, S1, DASH[0])):
        Ns = np.logspace(7, 12, 300); Ls = Lf(Ns, C / (6 * Ns)); k = int(np.argmin(Ls)); mins.append((Ns[k], Ls[k]))
        ax.plot(Ns, Ls, color=c, ls=ls); ax.annotate(f"C = {C:.0e} FLOPs", (Ns[k], Ls[k]), xytext=(0, 9), textcoords="offset points", ha="center", color=INK2, fontsize=7.5)
    mx, my = zip(*mins); ax.plot(mx, my, color=INK2, lw=0.8, ls=":", marker="o", ms=3.5)
    ax.text(1.2e7, 3.72, "dotted: compute-optimal size\nfor each budget", color=INK2, fontsize=7.5)
    ax.set_xscale("log"); ax.set_ylim(1.8, 4.0); ax.set_xlabel("model parameters N (tokens D = C / 6N)"); ax.set_ylabel("predicted loss")
    ax.set_title("IsoFLOP curves: at fixed compute, too small and too big both lose")
    save(fig, "13-fig-1.png")

    # 13-fig-2: best model size as lifetime inference grows (same calculation as code/ch13_check.py)
    G = (al * A / (be * B)) ** (1 / (al + be)); N0 = G * (1e23 / 6) ** (be / (al + be)); target = Lf(N0, 1e23 / (6 * N0))
    infs = np.logspace(10, 14.5, 40); bestN, ratio = [], []
    for I in infs:
        Ns = np.logspace(9.3, 11, 600); ok = target - E - A / Ns ** al > 0; Ns = Ns[ok]
        D = (B / (target - E - A / Ns ** al)) ** (1 / be); cost = 6 * Ns * D + 2 * Ns * I
        k = int(np.argmin(cost)); bestN.append(Ns[k] / 1e9); ratio.append(D[k] / Ns[k])
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.plot(infs, bestN, color=S1)
    for I in (1e12, 1e13, 1e14):
        k = int(np.argmin(abs(infs - I))); ax.annotate(f"{bestN[k]:.1f}B params,\n{ratio[k]:,.0f} tokens/param", (infs[k], bestN[k]), xytext=(6, 6),
                                                          textcoords="offset points", ha="left", color=INK2, fontsize=7)
    ax.set_xscale("log"); ax.set_ylim(0, 19); ax.set_xlabel("lifetime inference tokens"); ax.set_ylabel("best model size (B)")
    ax.set_title("Same target loss: more serving means a smaller model trained longer")
    save(fig, "13-fig-2.png")

    # 13-fig-3: smooth per-token gains look like sudden emergence under exact match
    p = np.linspace(0.3, 0.995, 200)
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.plot(p, p, color=S1); ax.plot(p, p ** 10, color=S2, ls=DASH[1])
    ax.annotate("per-token accuracy", (0.55, 0.55), xytext=(-6, 8), textcoords="offset points", ha="right", color=INK2, fontsize=8)
    ax.annotate("exact match, 10-token answer", (0.85, 0.85 ** 10), xytext=(-8, 6), textcoords="offset points", ha="right", color=INK2, fontsize=8)
    ax.set_xlim(0.3, 1.0); ax.set_xlabel("per-token accuracy (improves smoothly with scale)"); ax.set_ylabel("score")
    ax.set_title("The 'emergent' jump comes from the all-or-nothing metric")
    save(fig, "13-fig-3.png")


def ch14():
    # 14-fig-1: vision tokens vs image side for 14-px patches, with and without 2x2 merging
    side = np.arange(224, 1800, 16)
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.plot(side, (side // 14) ** 2, color=S2, ls=DASH[1]); ax.plot(side, (side // 28) ** 2, color=S1)
    label_end(ax, side[-1], (side[-1] // 14) ** 2, "14-px patches", S2); label_end(ax, side[-1], (side[-1] // 28) ** 2, "after 2×2 merging", S1)
    for s_ in (448, 896, 1344):
        ax.axvline(s_, color=GRID, lw=0.8)
    ax.set_xlim(200, 2300); ax.set_xlabel("image side (pixels)"); ax.set_ylabel("vision tokens")
    ax.set_title("Tokens grow with the square of resolution")
    save(fig, "14-fig-1.png")

    # 14-fig-2: diffusion forward process: signal and noise weights over time (linear beta schedule)
    betas = np.linspace(1e-4, 0.02, 1000); ab = np.cumprod(1 - betas); t = np.arange(1000)
    fig, ax = plt.subplots(figsize=(W, 2.1))
    ax.plot(t, np.sqrt(ab), color=S1); ax.plot(t, np.sqrt(1 - ab), color=S2, ls=DASH[1])
    label_end(ax, 999, np.sqrt(ab)[-1], "signal weight √ᾱ", S1); label_end(ax, 999, np.sqrt(1 - ab)[-1], "noise weight √(1 − ᾱ)", S2)
    ax.set_xlim(0, 1350); ax.set_xlabel("diffusion step t"); ax.set_ylabel("weight in x_t")
    ax.set_title("Forward process: the image fades into noise over 1,000 steps")
    save(fig, "14-fig-2.png")

# ---------------------------------------------------------------------------------------------
def ch15():
    logits = np.array([6.0, 3.5, 3.0, 1.0, 0.5, 0.0]); names = ["shawarma", "falafel", "kebab", "pizza", "sushi", "salad"]
    fig, ax = plt.subplots(figsize=(W, 2.2)); x = np.arange(6); wdt = 0.26
    for k, (T, c) in enumerate(((1, S1), (2, S2), (5, S3))):
        p = np.exp(logits / T); p /= p.sum()
        ax.bar(x + (k - 1) * wdt, p, width=wdt - 0.03, color=c, label=f"T = {T}")
        print(f"T={T}:", np.round(p, 3).tolist())
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("teacher probability"); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.5, loc="upper right"); ax.set_title("Higher temperature reveals what the teacher thinks is similar")
    save(fig, "15-fig-1.png")


def ch16():
    s_ = np.linspace(0, 1, 300)
    fig, ax = plt.subplots(figsize=(W, 2.2))
    for (b, r), c, ls in (((16, 8), S1, DASH[0]), ((20, 5), S2, DASH[1]), ((32, 4), S3, DASH[2])):
        p = 1 - (1 - s_ ** r) ** b; thr = (1 / b) ** (1 / r)
        ax.plot(s_, p, color=c, ls=ls, label=f"{b} bands × {r} rows (threshold ≈ {thr:.2f})")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlabel("Jaccard similarity of two documents"); ax.set_ylabel("P(flagged as candidates)")
    ax.set_title("LSH turns similarity into a tunable S-curve")
    save(fig, "16-fig-1.png")

    ep = np.arange(1, 41); U = 100.0; R = ep - 1
    eff = U + U * 15.4 * (1 - np.exp(-R / 15.4))
    fig, ax = plt.subplots(figsize=(W, 2.1))
    ax.plot(ep, ep * U / 1000, color=MUTED, ls=DASH[1]); ax.plot(ep, eff / 1000, color=S1)
    ax.text(22, 2.75, "tokens seen (if repeats were\nas good as new data)", color=INK2, fontsize=7.5, ha="right")
    ax.annotate("effective tokens", (40, eff[-1] / 1000), xytext=(-4, 8), textcoords="offset points", ha="right", color=INK2, fontsize=8)
    for e in (4, 16):
        ax.annotate(f"{e} epochs: {eff[e-1]:.0f}B effective", (e, eff[e - 1] / 1000), xytext=(6, -14), textcoords="offset points", color=INK2, fontsize=7.5)
        ax.plot([e], [eff[e - 1] / 1000], "o", color=S1, ms=3)
    ax.set_xlabel("epochs over 100B unique tokens"); ax.set_ylabel("trillions of tokens"); ax.set_xlim(0, 41); ax.set_ylim(-0.35, 4.2)
    ax.set_title("Repeated data loses value after a few epochs")
    save(fig, "16-fig-2.png")


def ch17():
    fig, ax = plt.subplots(figsize=(W, 1.9))
    names = ["DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3 / FSDP"][::-1]; vals = [1120.0, 293.1, 155.3, 17.5][::-1]
    ax.barh(names, vals, color=S1, height=0.55); ax.axvline(80, color=S2, lw=0.9, ls=DASH[1])
    ax.text(88, 1.5, "80 GB GPU", color=INK2, fontsize=7.5, va="center")
    for y, v in enumerate(vals): ax.text(v + 12, y, f"{v:,.0f} GB" if v >= 20 else f"{v:.1f} GB", va="center", color=INK2, fontsize=8)
    ax.set_xlim(0, 1300); ax.grid(axis="y", visible=False); ax.set_xlabel("memory per GPU for weights, gradients and Adam states (70B over 64 GPUs)")
    ax.set_title("Only full sharding fits a 70B model's training state on 80 GB GPUs")
    save(fig, "17-fig-1.png")

    m = np.arange(1, 129)
    fig, ax = plt.subplots(figsize=(W, 2.1))
    for p, c, ls in ((4, S1, DASH[0]), (8, S2, DASH[1]), (16, S3, DASH[2])):
        b = (p - 1) / (m + p - 1); ax.plot(m, b * 100, color=c, ls=ls)
        k = 7; ax.annotate(f"{p} stages", (m[k], b[k] * 100), xytext=(5, 3), textcoords="offset points", color=INK2, fontsize=7.5)
    ax.axhline(15, color=MUTED, lw=0.7, ls=":"); ax.text(100, 17, "15%", color=MUTED, fontsize=7.5)
    ax.set_xscale("log", base=2); ax.set_xlabel("micro-batches per batch"); ax.set_ylabel("idle time (%)")
    ax.set_title("The pipeline bubble shrinks with more micro-batches")
    save(fig, "17-fig-2.png")


def ch19():
    rng = np.random.default_rng(0)
    p = np.linspace(0, 1, 300)
    fig, ax = plt.subplots(figsize=(W, 2.1))
    for G, c, ls in ((4, S2, DASH[1]), (8, S1, DASH[0]), (16, S3, DASH[2])):
        z = p ** G + (1 - p) ** G; ax.plot(p, z, color=c, ls=ls)
        ax.annotate(f"group of {G}", (0.12 if G == 4 else (0.16 if G == 8 else 0.2), (lambda q: q ** G + (1 - q) ** G)(0.12 if G == 4 else (0.16 if G == 8 else 0.2))),
                    xytext=(7, 5), textcoords="offset points", color=INK2, fontsize=7.5)
    ax.set_xlabel("prompt pass rate under the current policy"); ax.set_ylabel("P(no learning signal)")
    ax.set_title("Groups that are all right or all wrong teach GRPO nothing")
    save(fig, "19-fig-1.png")

    # pass@k before and after RL in the tabular simulation of code/ch19_check.py
    J, S_ = 200, 6; q = np.zeros((J, S_)); q[:, 0] = rng.uniform(0.3, 0.9, J); q[:, 0] *= rng.random(J) > 0.25
    q[:, 1:] = rng.uniform(0, 0.05, (J, S_ - 1)); hard = q[:, 0] == 0; q[hard, 5] = 0.8
    logits = np.log(np.array([0.70, 0.12, 0.08, 0.05, 0.04, 0.01]))[None].repeat(J, 0); theta = np.zeros(S_)
    def policy():
        z = logits + theta; z -= z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
    base = policy()
    for step in range(400):
        js = rng.integers(0, J, 32); grad = np.zeros(S_); pi = policy()
        for j in js:
            acts = rng.choice(S_, 8, p=pi[j]); r = (rng.random(8) < q[j, acts]).astype(float)
            if r.std() == 0: continue
            A_ = (r - r.mean()) / r.std()
            for a, adv in zip(acts, A_):
                g = -pi[j].copy(); g[a] += 1; grad += adv * g
        theta += 0.05 * grad / len(js)
    after = policy(); ks = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    pk = lambda pi: [(1 - (1 - (pi * q).sum(1)) ** k).mean() for k in ks]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.plot(ks, pk(base), color=S2, ls=DASH[1], marker="o", ms=3); ax.plot(ks, pk(after), color=S1, marker="o", ms=3)
    label_end(ax, ks[-1], pk(base)[-1], "base policy", S2); label_end(ax, ks[-1], pk(after)[-1], "after RL", S1)
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks); ax.set_xlim(0.8, 900)
    ax.set_xlabel("k (samples per problem)"); ax.set_ylabel("pass@k"); ax.set_title("RL wins at k = 1 and loses at large k: sharpening, not new solutions")
    save(fig, "19-fig-2.png")

    from collections import Counter
    def maj(pc, n, wrong=5, trials=4000):
        w = 0
        for _ in range(trials):
            ans = np.where(rng.random(n) < pc, 0, rng.integers(1, wrong + 1, n)); c = Counter(ans.tolist()); top = max(c.values())
            win = [a for a, v in c.items() if v == top]; w += (0 in win) / len(win)
        return w / trials
    ns = [1, 3, 5, 9, 15, 25, 41]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    for (pc, wrong, lab), c, ls in (((0.5, 5, "50% correct, errors spread"), S1, DASH[0]), ((0.3, 5, "30% correct, errors spread"), S3, DASH[2]),
                                    ((0.3, 1, "30% correct, one popular wrong answer"), S2, DASH[1])):
        ys = [maj(pc, n, wrong) for n in ns]; ax.plot(ns, ys, color=c, ls=ls, marker="o", ms=3); label_end(ax, ns[-1], ys[-1], lab, c)
    ax.set_xlim(0, 75); ax.set_xlabel("samples voted on"); ax.set_ylabel("accuracy of the majority answer")
    ax.set_title("Voting amplifies whatever answer the model believes most")
    save(fig, "19-fig-3.png")


def ch20():
    import torch, torch.nn as nn
    torch.manual_seed(0)
    # pretrain a small MLP on task A, fine-tune fully on a related task B, look at the update's singular values
    d = 64; Wt = torch.randn(d, d) / 8
    XA = torch.randn(4000, d); yA = torch.tanh(XA @ Wt) @ torch.randn(d, 1)
    net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(1500):
        loss = (net(XA) - yA).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    W0 = net[2].weight.detach().clone()
    yB = yA + 0.5 * torch.relu(XA[:, :3].sum(1, keepdim=True))          # a related task with a small new component
    opt = torch.optim.Adam(net.parameters(), 3e-4)
    for _ in range(800):
        loss = (net(XA) - yB).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    dW = net[2].weight.detach() - W0
    sv = torch.linalg.svdvals(dW).numpy(); energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    rnd = torch.linalg.svdvals(torch.randn(256, 256)).numpy(); er = np.cumsum(rnd ** 2) / np.sum(rnd ** 2)
    fig, ax = plt.subplots(figsize=(W, 2.2)); r = np.arange(1, 257)
    ax.plot(r, energy, color=S1); ax.plot(r, er, color=MUTED, ls=DASH[1])
    for k in (4, 16): ax.annotate(f"rank {k}: {energy[k-1]:.0%}", (k, energy[k - 1]), xytext=(6, -10), textcoords="offset points", color=INK2, fontsize=7.5)
    ax.annotate("random 256×256 matrix", (120, er[119]), xytext=(6, -12), textcoords="offset points", color=INK2, fontsize=7.5)
    ax.annotate("fine-tuning update ΔW", (1, energy[0]), xytext=(8, -2), textcoords="offset points", color=INK2, fontsize=7.5)
    ax.set_xscale("log", base=2); ax.set_xlabel("rank kept"); ax.set_ylabel("share of ΔW's energy captured")
    ax.set_title("A fine-tuning update is approximately low rank")
    print(f"fine-tune update energy: rank4 {energy[3]:.3f} rank16 {energy[15]:.3f}; random rank16 {er[15]:.3f}")
    save(fig, "20-fig-1.png")


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["10", "18"]
    for ch in wanted:
        globals()[f"ch{ch}"]()

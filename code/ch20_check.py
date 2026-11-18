import math, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r, self.scaling = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if not self.merged:
            out = out + (self.dropout(x) @ self.A.T @ self.B.T) * self.scaling
        return out

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += (self.B @ self.A) * self.scaling
            self.merged = True

def count(model):
    total = sum(p.numel() for p in model.parameters()); train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train, total

base = nn.Linear(4096, 4096, bias=False)
lora = LoRALinear(base, r=16, alpha=32)
print(count(lora), round(count(lora)[0] / count(lora)[1] * 100, 2))

# toy adaptation: base learns task A; LoRA adapts to task B with a low-rank change
d = 64
base = nn.Linear(d, d, bias=False)
W_target = base.weight.detach() + (torch.randn(d, 4) @ torch.randn(4, d)) * 0.1
X = torch.randn(2048, d); Y = X @ W_target.T
model = LoRALinear(base, r=4, alpha=8)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
print("before", round(F.mse_loss(model(X), Y).item(), 4))
for step in range(500):
    loss = F.mse_loss(model(X), Y); opt.zero_grad(); loss.backward(); opt.step()
print("after", round(loss.item(), 6))
x = torch.randn(3, d); y1 = model(x); model.merge(); y2 = model(x)
print("merge equal", torch.allclose(y1, y2, atol=1e-5))

# prompt tuning: soft prompt prepended to frozen embeddings
class SoftPrompt(nn.Module):
    def __init__(self, n_tokens, d_model, init_from=None):
        super().__init__()
        init = init_from[:n_tokens].clone() if init_from is not None else torch.randn(n_tokens, d_model) * 0.02
        self.prompt = nn.Parameter(init)
    def forward(self, token_embeds):
        B = token_embeds.shape[0]
        return torch.cat([self.prompt.unsqueeze(0).expand(B, -1, -1), token_embeds], dim=1)
emb = nn.Embedding(100, 32); sp = SoftPrompt(10, 32, init_from=emb.weight.detach())
print(sp(emb(torch.randint(0, 100, (2, 7)))).shape)

def finetune_memory_gb(n_params_b, mode):
    n = n_params_b * 1e9
    if mode == "full_adam_mixed":
        return n * 16 / 1e9          # bf16 weights+grads (4) + fp32 master (4) + Adam m,v (8)
    if mode == "lora_bf16":
        return n * 2 / 1e9
    if mode == "qlora_4bit":
        return n * 0.5 / 1e9
for m in ["full_adam_mixed", "lora_bf16", "qlora_4bit"]:
    print(m, finetune_memory_gb(7, m))

import numpy as np
# 20.4 NF4 vs uniform 4-bit on normal weights, block size 64, absmax scaling
NF4 = np.array([-1, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1])
w = np.random.default_rng(0).normal(0, 0.02, (20000, 64)); amax = np.abs(w).max(1, keepdims=True)
for name, lv in (("NF4", NF4), ("uniform int4", np.linspace(-1, 1, 16))):
    q = lv[np.abs((w / amax)[..., None] - lv).argmin(-1)] * amax
    print(f"{name}: relative RMS error {np.sqrt(((q - w) ** 2).mean()) / w.std():.3f}")
print("bits/param for constants: %.3f plain, %.3f double-quantised" % (32 / 64, 8 / 64 + 32 / (64 * 256)))
# 20.12 LoRA r=16 on all linear layers of a 7B Llama-style model (32 layers, d=4096, MLP 11008)
lp = 32 * (4 * 16 * (4096 + 4096) + 3 * 16 * (4096 + 11008))
print(f"LoRA params {lp/1e6:.1f}M ({lp/6.74e9:.2%}); adapter weights+grads+Adam {lp*16/1e9:.2f} GB; 4-bit base {6.74e9*0.5/1e9:.2f} GB")

import os, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(0)

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x ** 2).sum() + 3 * x[0]
y.backward(); print(x.grad)

class OrdersDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32); self.y = torch.as_tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class ReorderModel(nn.Module):
    def __init__(self, d_in, d_hidden=64, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(d_hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

rng = np.random.default_rng(0)
X = rng.normal(size=(4000, 10)); w = rng.normal(size=10)
yv = (rng.random(4000) < 1 / (1 + np.exp(-(X @ w)))).astype(np.float32)
train_ds, val_ds = OrdersDataset(X[:3200], yv[:3200]), OrdersDataset(X[3200:], yv[3200:])
g = torch.Generator().manual_seed(0)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, generator=g, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=512)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ReorderModel(10).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=20 * len(train_dl))

def evaluate(model, dl):
    model.eval(); total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            total += F.binary_cross_entropy_with_logits(model(xb), yb, reduction="sum").item(); n += len(yb)
    return total / n

best = float("inf")
for epoch in range(20):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    val = evaluate(model, val_dl)
    if val < best:
        best = val
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "val": val}, "/tmp/best.pt")
print("best val loss", round(best, 4))
ckpt = torch.load("/tmp/best.pt"); m2 = ReorderModel(10); m2.load_state_dict(ckpt["model"]); print("loaded epoch", ckpt["epoch"])

# gradient accumulation
accum = 4; opt.zero_grad()
for i, (xb, yb) in enumerate(train_dl):
    loss = F.binary_cross_entropy_with_logits(model(xb), yb) / accum
    loss.backward()
    if (i + 1) % accum == 0:
        opt.step(); opt.zero_grad()
    if i > 8: break
print("accum ok")

# autocast on cpu bf16
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    out = model(torch.randn(4, 10))
print(out.dtype)

# ---- review additions ---------------------------------------------------------------------
# 6.1 autograd details: in-place error, leaf vs non-leaf, retain_graph, custom Function, hooks
a = torch.randn(3, requires_grad=True)
b = a.sigmoid()                 # sigmoid saves its output for backward
b.mul_(2)                       # in-place change of a saved tensor
try:
    b.sum().backward(); print("in-place: no error (unexpected)")
except RuntimeError as e:
    print("in-place error:", "modified by an inplace operation" in str(e))

x = torch.tensor([1.0, 2.0], requires_grad=True)
h = x * 3; h.retain_grad()
out = (h ** 2).sum(); out.backward(retain_graph=True)
print("leaf", x.is_leaf, "non-leaf", h.is_leaf, "h.grad", h.grad.tolist(), "x.grad", x.grad.tolist())
out.backward()                  # second backward works only because of retain_graph=True
print("x.grad after second backward (accumulated):", x.grad.tolist())

class ClampGrad(torch.autograd.Function):
    """Identity in the forward pass; clips the gradient to [-1, 1] in the backward pass."""
    @staticmethod
    def forward(ctx, inp):
        return inp.view_as(inp)
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.clamp(-1, 1)
z = torch.tensor([1.0, -2.0], requires_grad=True)
(ClampGrad.apply(z) * torch.tensor([5.0, 0.3])).sum().backward()
print("custom Function grad:", z.grad.tolist())

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
grad_norms = {}
for name, p in net.named_parameters():
    p.register_hook(lambda g, name=name: grad_norms.__setitem__(name, round(g.norm().item(), 4)))
acts = {}
net[1].register_forward_hook(lambda mod, inp, outp: acts.__setitem__("relu_frac_zero", (outp == 0).float().mean().item()))
torch.manual_seed(0); net(torch.randn(16, 4)).pow(2).mean().backward()
print("hooks logged", sorted(grad_norms), "relu zero fraction", round(acts["relu_frac_zero"], 3))

# 6.5 memory arithmetic: bf16 weights 2 + bf16 grads 2 + fp32 master 4 + Adam m 4 + Adam v 4
bytes_per_param = 2 + 2 + 4 + 4 + 4
print("bytes/param", bytes_per_param, " 7B model:", 7e9 * bytes_per_param / 1e9, "GB before activations")

# 6.6 resuming RNG state reproduces the next random draws exactly
state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
draw1 = (torch.rand(2).tolist(), np.random.rand(), random.random())
torch.set_rng_state(state["torch"]); np.random.set_state(state["numpy"]); random.setstate(state["python"])
draw2 = (torch.rand(2).tolist(), np.random.rand(), random.random())
print("RNG restore reproduces draws:", draw1 == draw2)

# 6.8 accumulation with variable token counts: mean-of-means vs sum / total tokens
torch.manual_seed(0)
lin = nn.Linear(4, 3)
xs = [torch.randn(10, 4), torch.randn(90, 4)]          # micro-batch A: 10 tokens, B: 90 tokens
ys = [torch.randint(0, 3, (10,)), torch.randint(0, 3, (90,))]
def grad_of(fn):
    lin.zero_grad(); fn(); return lin.weight.grad.clone()
g_full = grad_of(lambda: F.cross_entropy(lin(torch.cat(xs)), torch.cat(ys)).backward())
g_naive = grad_of(lambda: [(F.cross_entropy(lin(x), y) / 2).backward() for x, y in zip(xs, ys)])
n_tok = sum(len(y) for y in ys)
g_fixed = grad_of(lambda: [(F.cross_entropy(lin(x), y, reduction="sum") / n_tok).backward() for x, y in zip(xs, ys)])
print("accumulation: naive matches full batch", torch.allclose(g_naive, g_full, atol=1e-6),
      " sum/total-tokens matches", torch.allclose(g_fixed, g_full, atol=1e-6))
print("toy numbers: mean of means", (2.0 + 1.0) / 2, " true per-token mean", (10 * 2.0 + 90 * 1.0) / 100)

# 6.10 MFU: 6 * N * tokens/s / peak. 1B params, 50k tokens/s, H100 SXM dense bf16 peak ~989 TFLOP/s
print("MFU example:", round(6 * 1e9 * 5e4 / 989e12, 3))

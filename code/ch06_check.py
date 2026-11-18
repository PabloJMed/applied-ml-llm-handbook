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

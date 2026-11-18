import torch, torch.nn.functional as F
torch.manual_seed(0)
# SFT label masking
prompt_ids = torch.tensor([5, 6, 7]); answer_ids = torch.tensor([8, 9, 2])
input_ids = torch.cat([prompt_ids, answer_ids]); labels = input_ids.clone(); labels[:len(prompt_ids)] = -100
logits = torch.randn(1, 6, 20)
loss = F.cross_entropy(logits[:, :-1].reshape(-1, 20), labels[1:].reshape(-1), ignore_index=-100)
print("sft loss", round(loss.item(), 3), "tokens scored", (labels[1:] != -100).sum().item())

def seq_logprob(logits, labels):
    lp = logits[:, :-1].log_softmax(-1); tgt = labels[:, 1:]
    mask = tgt != -100
    tok = lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (tok * mask).sum(-1)

# Bradley-Terry reward model loss
r_chosen, r_rejected = torch.tensor([1.2, 0.3]), torch.tensor([0.2, 0.5])
print("rm loss", round(-F.logsigmoid(r_chosen - r_rejected).mean().item(), 4))

# DPO
def dpo_loss(pc, pr, rc, rr, beta=0.1):
    logits = beta * ((pc - rc) - (pr - rr))
    return -F.logsigmoid(logits).mean(), (beta * (pc - rc)).mean(), (beta * (pr - rr)).mean()
pc, pr = torch.tensor([-12.0, -20.0]), torch.tensor([-15.0, -18.0]); rc, rr = torch.tensor([-13.0, -19.0]), torch.tensor([-14.0, -19.0])
l, cr, rj = dpo_loss(pc, pr, rc, rr); print("dpo", round(l.item(), 4), round(cr.item(), 3), round(rj.item(), 3))
print("dpo at init", round(dpo_loss(rc, rr, rc, rr)[0].item(), 4))

# GRPO group-relative advantages with verifiable reward
rewards = torch.tensor([1., 0., 0., 1., 1., 0., 0., 0.])
adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6); print("grpo adv", [round(a, 2) for a in adv.tolist()])

# KL-penalised reward for PPO-RLHF
def shaped_reward(rm_score, logp_policy, logp_ref, beta=0.05):
    return rm_score - beta * (logp_policy - logp_ref).sum()
print("shaped", round(shaped_reward(2.0, torch.tensor([-1., -0.5]), torch.tensor([-2., -1.5])).item(), 3))

# distillation loss
def distill_loss(student_logits, teacher_logits, y, T=2.0, alpha=0.5):
    kd = F.kl_div(F.log_softmax(student_logits / T, -1), F.softmax(teacher_logits / T, -1), reduction="batchmean") * T * T
    return alpha * kd + (1 - alpha) * F.cross_entropy(student_logits, y)
s, t = torch.randn(4, 5), torch.randn(4, 5); print("distill", round(distill_loss(s, t, torch.tensor([0,1,2,3])).item(), 3), round(distill_loss(t, t, torch.tensor([0,1,2,3]), alpha=1).item(), 4))

# 15.9 compute of each stage for a 7B model (6N FLOPs per trained token, 2N per forward-only or generated token)
N = 7e9; pre = 6 * N * 2e12
sft = 6 * N * 100_000 * 1000 * 3; dpo = (6 + 2) * N * 50_000 * 2 * 1000
rl_tok = 1000 * 512 * 8 * 2000; rl = (2 + 6) * N * rl_tok
for k, v in [("pre-training", pre), ("SFT", sft), ("DPO", dpo), ("RLVR", rl)]:
    print(f"{k}: {v:.2e} FLOPs = {v / pre:.4%} of pre-training")

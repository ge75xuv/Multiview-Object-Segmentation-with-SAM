import torch, torch.nn as nn, torch.nn.functional as F

class ViewIDFiLM(nn.Module):
    def __init__(self, num_views, channels_per_scale, d_model=256, hid=64):
        super().__init__()
        self.emb = nn.Embedding(num_views, hid)
        self.film = nn.ModuleList([nn.Linear(hid, 2*c) for c in channels_per_scale])
        self.q_bias = nn.Linear(hid, d_model)
        self.mem_aff = nn.Linear(hid, 2*d_model)
        # safe zero init → starts as no-op
        for h in self.film: nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
        nn.init.zeros_(self.q_bias.weight); nn.init.zeros_(self.q_bias.bias)
        nn.init.zeros_(self.mem_aff.weight); nn.init.zeros_(self.mem_aff.bias)

    def forward(self, view_ids):
        h = self.emb(view_ids)                     # [B,hid]
        gammas, betas = [], []
        for head in self.film:
            g,b = head(h).chunk(2, -1)            # [B,C] x2
            gammas.append(g[:, :, None, None])    # [B,C,1,1]
            betas.append(b[:, :, None, None])
        q_bias = self.q_bias(h)[:, None, :]       # [B,1,d_model]
        ms, mb = self.mem_aff(h).chunk(2, -1)     # [B,d_model] x2
        return {"gamma": gammas, "beta": betas,
                "q_bias": q_bias,
                "mem_scale": ms[:, None, :], "mem_bias": mb[:, None, :]}

def apply_film(feats, cond):
    out = []
    for f, g, b in zip(feats, cond["gamma"], cond["beta"]):
        out.append(f * (1 + g) + b)
    return out


import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalGlobalMaskAwareTransformer(nn.Module):

    def __init__(
        self,
        T=300,
        d_model=48,
        nhead=3,
        num_layers=4,
        dim_ff=96,
        p_drop=0.45,
        num_classes=1,
        use_motion=False,
        num_global_tokens=1,
        local_window=8
        ):
        super().__init__()
        self.T = T
        self.use_motion = use_motion
        self.G = num_global_tokens
        self.w = local_window

        in_dim = 6 * (2 if use_motion else 1)
        self.input_proj = nn.Linear(in_dim, d_model)

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.glob_token = nn.Parameter(torch.zeros(1, self.G, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.G + self.T, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=p_drop,
            activation="gelu", batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        attn_mask = self._build_attn_mask()
        self.register_buffer("attn_mask", attn_mask, persistent=False)

        
    @staticmethod
    def make_valid_mask(x, mask_value=-50.0, eps=1e-6):
        miss = (x - mask_value).abs().sum(dim=1) < eps   # (B, T)

        valid = ~miss
        return valid

    @staticmethod
    def safe_diff(x, valid):
        # x: (B, 6, T), valid: (B, T)
        v = torch.zeros_like(x)
        if x.size(-1) > 1:
            dv = x[:, :, 1:] - x[:, :, :-1]
            both = (valid[:, 1:] & valid[:, :-1]).unsqueeze(1)  # (B,1,T-1)
            v[:, :, 1:] = dv * both
        return v

    def _build_attn_mask(self):

        S = 1 + self.G + self.T
        mask = torch.full((S, S), float('-inf'))
        idx_cls = 0
        idx_g0, idx_g1 = 1, self.G           # [1, G]
        idx_f0, idx_f1 = 1 + self.G, self.G + self.T  # [G+1, G+T]

        mask[idx_cls, :] = 0.0
        if self.G > 0:
            mask[idx_g0:idx_g1+1, :] = 0.0

        for i in range(idx_f0, idx_f1 + 1):
            j0 = max(idx_f0, i - self.w)
            j1 = min(idx_f1, i + self.w)
            mask[i, j0:j1+1] = 0.0
            mask[i, idx_cls] = 0.0
            if self.G > 0:
                mask[i, idx_g0:idx_g1+1] = 0.0

        return mask

    def forward(self, x):
        B, C, T = x.shape
        assert T == self.T, f"시퀀스 길이 {self.T}로 고정된 모델입니다. (입력 T={T})"

        valid = self.make_valid_mask(x)  # (B, T) True==유효, False==mask

        if self.use_motion:
            v = self.safe_diff(x, valid)   # (B,6,T)
            feat = torch.cat([x, v], dim=1)  # (B,12,T)
        else:
            feat = x                        # (B,6,T)

        feat = feat.transpose(1, 2)            # (B, T, C)
        feat = self.input_proj(feat)           # (B, T, d)

        mask_tok = self.mask_token.expand(B, T, -1)   # (B,T,d)
        feat = torch.where(valid.unsqueeze(-1), feat, mask_tok)

        cls  = self.cls_token.expand(B, 1, -1)        # (B,1,d)
        glb  = self.glob_token.expand(B, self.G, -1)  # (B,G,d)
        seq  = torch.cat([cls, glb, feat], dim=1)     # (B, 1+G+T, d)
        seq  = seq + self.pos_embed[:, :seq.size(1), :]

        if self.G > 0:
            pad = torch.zeros(B, 1 + self.G, dtype=torch.bool, device=x.device)
        else:
            pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([pad, ~valid], dim=1)  # (B, 1+G+T)

        enc = self.encoder(
            seq,
            mask=self.attn_mask,                      # (S,S) float
            src_key_padding_mask=key_padding_mask     # (B,S) bool
        )                                             # (B,S,d)

        cls_out = enc[:, 0]                 # (B,d)
        logits = self.head(self.norm(cls_out)).squeeze(-1)  # (B,)

        return logits

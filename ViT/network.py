import torch.nn as nn
import torch
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class MSA(nn.Module):
    def __init__(self, hidden_dim, heads, head_dim=64, dropout=0.):
        super(MSA,self).__init__()
        self.hidden_dim = head_dim * heads
        self.heads = heads
        self.head_dim = head_dim
        self.make_qkv = nn.Linear(hidden_dim, self.hidden_dim * 3, bias=False)
        self.softmax_layer = nn.Softmax(dim=-1)
        self.U_msa = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=0.1)
        )
        self.drop1 = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5
    def forward(self,x):
        batch, _, _ = x.shape
        # q, k, v = self.make_qkv(x).chunk(3, dim=-1)
        # Q, K, V = rearrange(q,'b n (h d) -> b h n d', h=self.heads), rearrange(k,'b n (h d) -> b h n d',h=self.heads), rearrange(v,'b n (h d) -> b h n d',h=self.heads)
        qkv = self.make_qkv(x).chunk(3, dim=-1)
        Q, K, V = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # denominator = math.sqrt(self.head_dim)
        scaled = torch.matmul(Q, K.transpose(-1,-2)) * self.scale
        attention_weight_out = self.softmax_layer(scaled)  # [batch,12,65,65]
        attention_weight = self.drop1(attention_weight_out)
        self_attention = torch.matmul(attention_weight, V)  # [batch,12,65,64]
        out = rearrange(self_attention, 'b h n d -> b n (h d)')
        out = self.U_msa(out)
        return out

class MLP(nn.Module):
    def __init__(self, dim, mlp_dim, dropout = 0.):
        super(MLP,self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.mlp(x)

class Transformer(nn.Module):
    def __init__(self, hidden_dim, heads, mlp_dim, layers, dropout=0.):
        super(Transformer, self).__init__()
        self.Layer = nn.ModuleList([])
        for _ in range(layers):
            self.Layer.append(nn.ModuleList([
                nn.LayerNorm(hidden_dim), #[batch,65,768]
                MSA(hidden_dim, heads, dropout=dropout), #[batch,65,768]
                nn.LayerNorm(hidden_dim),
                MLP(hidden_dim, mlp_dim, dropout=dropout) #[batch,65,768]
            ]))

    def forward(self,x):
        for ln1, msa, ln2, mlp in self.Layer:
            x = msa(ln1(x)) + x
            x = mlp(ln2(x)) + x
        return x

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, hidden_dim, heads, mlp_dim, layers, dropout):
        super(ViT,self).__init__()

        #use pre-trained model
        # trained_pos_emb = pre_model.encoder.pos_embedding[:, 1:, :]  # [1,196,768]
        # H = W = int(trained_pos_emb.shape[1] ** 0.5)
        # devide_pos_emb = trained_pos_emb.reshape(1, H, W, 768).permute(0, 3, 1, 2)
        # inter_pos_emb = nn.functional.interpolate(devide_pos_emb, size=(8, 8), mode='nearest')
        # new_pos_emb = inter_pos_emb.permute(0, 3, 1, 2).reshape(1, -1, 768)
        # self.pos_emb_param = torch.cat([pre_model.encoder.pos_embedding[:, 0, :].reshape(1, 1, 768), new_pos_emb], dim=1)
        # self.cls_Emb_param = pre_model.class_token

        num_patchs = (image_size // patch_size) ** 2 #64
        token_length = patch_size ** 2 * 3 #768
        self.patch_size = patch_size #16
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size), #[batch,64,768]
            nn.LayerNorm(token_length),
            nn.Linear(token_length, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.cls_Emb_param = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_emb_param = nn.Parameter(torch.randn(1, num_patchs + 1, hidden_dim))
        self.drop = nn.Dropout(dropout)

        # self.flatten_patch = Embedded_patch(image_size, patch_size, hidden_dim)
        self.transformer = Transformer(hidden_dim, heads, mlp_dim, layers, dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 200)
        )
    def forward(self,x):
        out = self.patch_embedding(x) #[batch,64,768]
        batch, n, _ = out.shape
        cls_token = repeat(self.cls_Emb_param, '1 1 d -> b 1 d', b=batch) #[batch,65,768]
        out = torch.cat((cls_token, out), dim=1)  #[batch,65,768]
        out += self.pos_emb_param[:, :(n + 1)]
        out = self.drop(out)
        # out = self.flatten_patch(x)
        out = self.transformer(out)

        # out = self.mlp_head(out[:,0,:])
        return out[:,0:,:]


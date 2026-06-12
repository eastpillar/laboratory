import torch.nn as nn
import torch
from einops import rearrange, repeat
from network_ViT2 import VisionTransformer

class MSA(nn.Module):
    def __init__(self, hidden_dim, heads, head_dim=16, dropout=0.):
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
            nn.Tanh(),
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

class TokenHPE(nn.Module):
    def __init__(self, ViT, image_size=224, patch_size=14, heads=8, mlp_dim=3072, layers=12, dropout=0.1, token_dim=768, dim=128, ori_tokens=9):
        super(TokenHPE,self).__init__()
        self.num_ori_token = ori_tokens
        # self.ViT = ViT
        self.feature_extractor = VisionTransformer(img_size = 224,
                                    patch_size = 16,
                                    embed_dim = 768,
                                    num_heads = 12,
                                    depth = 12,
                                    mlp_head=False)
        self.feature_embedding = nn.Linear(token_dim, dim)
        self.pos_emb = nn.Parameter(self.sine_pos_embedding(patch_size, patch_size, dim), requires_grad=False)
        self.ori_tokens = nn.Parameter(torch.zeros(1, ori_tokens, dim))
        self.transformer = Transformer(dim, heads, mlp_dim, layers, dropout)
        self.dir_token = nn.Linear(dim,6)

        self.mlp_head = nn.Sequential(
            nn.Linear(self.num_ori_token*9, self.num_ori_token*27),
            nn.Tanh(),
            nn.Linear(self.num_ori_token*27, 6)
        )

    def sine_pos_embedding(self, patch_h, patch_w, dim):
        epsilon = 0.0000001
        self.h, self.w = patch_h, patch_w
        ones = torch.ones(1, patch_h, patch_w)
        ones_x = ones.cumsum(2, dtype=torch.float32)
        ones_y = ones.cumsum(1, dtype=torch.float32)
        half_dim = dim // 2
        dim_i = torch.arange(half_dim, dtype=torch.float32)
        scale_x = (ones_x / ones_x[:, :, -1:] + epsilon) * (2 * torch.pi)
        scale_y = (ones_y / ones_y[:, -1:, :] + epsilon) * (2 * torch.pi)
        emb_equ = 10000 ** ((2 * dim_i) / dim)
        scale_x = scale_x[:, :, :, None] / emb_equ
        scale_y = scale_y[:, :, :, None] / emb_equ

        sincos_x = torch.stack((scale_x[:, :, :, 0::2].sin(), scale_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        sincos_y = torch.stack((scale_y[:, :, :, 0::2].sin(), scale_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_emb = torch.cat((sincos_x, sincos_y), dim=3).permute(0, 3, 1, 2)
        pos_emb = pos_emb.flatten(2).permute(0, 2, 1)

        return pos_emb

    def matrix_6d(self, x):
        x_axis = x[:, 0:3]
        y_axis = x[:, 3:6]
        sum_x = torch.sqrt(x_axis.pow(2).sum(1))
        sum_x = sum_x.view(sum_x.shape[0], 1).expand(sum_x.shape[0], x_axis.shape[1])
        X = x_axis / sum_x

        i = x_axis[:, 1] * y_axis[:, 2] - x_axis[:, 2] * y_axis[:, 1]
        j = x_axis[:, 2] * y_axis[:, 0] - x_axis[:, 0] * y_axis[:, 2]
        k = x_axis[:, 0] * y_axis[:, 1] - x_axis[:, 1] * y_axis[:, 0]
        out = torch.cat((i.view(x_axis.shape[0], 1), j.view(x_axis.shape[0], 1), k.view(x_axis.shape[0], 1)), 1)

        sum_x2 = torch.sqrt(out.pow(2).sum(1))
        sum_x2 = sum_x2.view(out.shape[0], 1).expand(out.shape[0], out.shape[1])
        Z = out / sum_x2

        i2 = out[:, 1] * x_axis[:, 2] - out[:, 2] * x_axis[:, 1]
        j2 = out[:, 2] * x_axis[:, 0] - out[:, 0] * x_axis[:, 2]
        k2 = out[:, 0] * x_axis[:, 1] - out[:, 1] * x_axis[:, 0]
        Y = torch.cat((i2.view(out.shape[0], 1), j2.view(out.shape[0], 1), k2.view(out.shape[0], 1)), 1)

        X, Y, Z = X.view(-1, 3, 1), Y.view(-1, 3, 1), Z.view(-1, 3, 1)
        matrix = torch.cat((X, Y, Z), dim=2)

        return matrix

    def forward(self,x):
        # x = self.ViT._process_input(x)
        # n = x.shape[0]
        #
        # batch_class_token = self.ViT.class_token.expand(n, -1, -1)
        # x = torch.cat([batch_class_token, x], dim=1)
        #
        # x = self.ViT.encoder(x)

        x = self.feature_embedding(x[:,1:,:]) + self.pos_emb
        x = torch.cat((self.ori_tokens, x), dim=1)
        x = self.transformer(x)
        x = self.dir_token(x[:,:self.num_ori_token,:])

        x = x.view(-1, x.shape[2])
        matrix = self.matrix_6d(x)

        matrix = matrix.view(1,self.num_ori_token, 3, 3)
        out = rearrange(matrix, 'b t d1 d2 -> b (t d1 d2)')

        out = self.mlp_head(out)
        pred = self.matrix_6d(out)
        return pred, matrix


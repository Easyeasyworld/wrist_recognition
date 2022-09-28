import torch
from torch import nn


class PatchEmbed(nn.Module):

    def __init__(self, input_dim=6, embed_dim=512):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, X):
        X = self.proj(X)
        X = self.norm(X)
        return X


class Attention(nn.Module):
    def __init__(self, dim, num_head, qkv_bias=False, qk_scal=None,
                 attn_drop_ratio=0., proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_head
        head_dim = dim // num_head
        self.scale = qk_scal or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, X):
        B, T, C = X.shape
        # qkv : b,t,c -> b,t,3c
        # reshape: b,t,3c -> b,t,3,num_head,c/num_head
        # permute :   b,t,3, num_head, C / num_head  ->    3, b, num_head, t, C/num_head
        qkv = self.qkv(X).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: -> [batch_size, num_head,dim_per_head,t]
        # @: multiply -> [batch_size, num_head, t,t]

        attn_weight = (q @ k.transpose(-2, -1)) * self.scale
        attn_weight = attn_weight.softmax(dim=-1)
        attn_weight = self.attn_drop(attn_weight)

        # @:bmm -> [batch_size,num_head,t,dim_per_head]
        # transpose->[batch_size,t,num_head,dim_per_head]
        # reshape ->[batch_size,t,dim]
        X = (attn_weight @ v).transpose(1, 2).reshape(B, T, C)
        X = self.proj(X)
        X = self.proj_drop(X)
        return X


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ratio=4, act_layer=nn.GELU, mlp_drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * ratio
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(mlp_drop)

    def forward(self, X):
        X = self.fc1(X)
        X = self.act(X)
        X = self.drop(X)
        X = self.fc2(X)
        X = self.drop(X)
        return X


class Block(nn.Module):
    def __init__(self, dim, num_head, qkv_bias=False, qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., act_layer=nn.GELU, **kwargs):
        super(Block, self).__init__(**kwargs)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_head, qkv_bias, qk_scale, attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, act_layer=act_layer, mlp_drop=drop_ratio)

    def forward(self, X):
        X = X + self.attn(self.norm1(X))
        X = X + self.mlp(self.norm2(X))
        return X


# embed_layer=PatchEmbed, norm_layer=None,mlp_ratio=4.0,
class CifTransformer(nn.Module):
    def __init__(self, input_dim=6, dim=12, num_classes=17, sam_rate=409,
                 depth=2, num_heads=4, qkv_bias=True,
                 qk_scale=None, drop_ratio=0., attn_drop_ratio=0.,
                 act_layer=None, **kwargs):
        super(CifTransformer, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.num_features = dim
        act_layer = act_layer or nn.GELU

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.embed = PatchEmbed(input_dim, dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, sam_rate + 1, dim))  # 401T+1
        self.pos_drop = nn.Dropout(drop_ratio)

        self.blocks = nn.Sequential(
            *[Block(dim=dim, num_head=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop_ratio=attn_drop_ratio, drop_ratio=drop_ratio, act_layer=act_layer)
              for i in range(depth)]
        )

        self.norm = nn.LayerNorm(dim)

        # classfier head
        self.head = nn.Linear(self.num_features, self.num_classes)

    def forward_feature(self, X):
        # B,T,6 -->B,T,512
        X = self.embed(X)
        # cls_token: 1,1,512 --> B,1,512
        cls_token = self.cls_token.expand(X.shape[0], -1, -1)
        # X --> B,1+T，512
        X = torch.cat((cls_token, X), dim=1)
        # 位置信息
        # print(self.pos_embed.shape)
        # print(X.shape)
        X = X + self.pos_embed

        X = self.blocks(X)
        # 归一化
        X = self.norm(X)
        return X

    def forward(self, X):
        X = self.forward_feature(X)
        X = self.head(X[:, 0])
        return X


# input_dim  6通道数据
# dim = 512  embed维度
# num_classes = 17  总类别
# depth = 8         transformer_layer
# num_heads = 8     注意力头
# qkv_bias = True
# qk_scale = None   防止过大
# drop_ratio = 0.   全部fc失活
# attn_drop_ratio = 0. attn失活
# act_layer = None   激活函数

# data = torch.zeros(2, 2, 6)
# print(data)
#
# input_dim, dim, num_classes, depth, num_heads = 6, 512, 17, 8, 8
# qkv_bias, qk_scale, attn_drop_ratio, drop_ratio, act_layer = True, None, 0., 0., None
#
# net = CifTransformer(input_dim, dim, num_classes, depth, num_heads,
#                      qkv_bias, qk_scale, drop_ratio, attn_drop_ratio, act_layer)
# y = net(data)
# print(y.shape)

def create_model(num_classes, sam_rate=401):
    model = CifTransformer(num_classes=num_classes, sam_rate=sam_rate)
    return model

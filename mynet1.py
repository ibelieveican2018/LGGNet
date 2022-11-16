import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
import numpy as np
import argparse
from einops.layers.torch import Rearrange
import math


def weights_init(m):
    #classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Conv3d):
         init.kaiming_normal_(m.weight)
         init.normal_(m.bias, std=1e-6)
    if isinstance(m, nn.Linear): 
        init.xavier_uniform_(m.weight) #torch.nn.init.xavier_uniform_(tensor, gain=1) 均匀分布 ~  U(−a,a)
        torch.nn.init.normal_(m.bias, std=1e-6)

class SS(nn.Module):  # nn.Module是在pytorch使用非常广泛的类，搭建网络基本都需要用到这个。
    def __init__(self):
        super().__init__()
        self.conv3d_spatial_features = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=12, kernel_size=(1,3, 3), padding=(0,1,1), bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv3d(in_channels=12, out_channels=12, kernel_size=(1,3, 3), padding=(0,1,1), bias=True),
            nn.BatchNorm3d(12),
            nn.Dropout(0.1)
        )

        self.conv3d_spectral_features = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=12, kernel_size=(3,1, 1), padding=(1,0,0), bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv3d(in_channels=12, out_channels=12, kernel_size=(3, 1,1), padding=(1,0,0), bias=True),
            nn.BatchNorm3d(12),
            nn.Dropout(0.1)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
         #######################################spatial_branch1###########################
        x_spatial = self.conv3d_spatial_features(x)  # x1 torch.Size([64, 16, 7, 36])
        x_spatial = x + x_spatial
        x_spatial= self.relu ( x_spatial)
        ######################################spectral_branch1###########################
        x_spectral = self.conv3d_spectral_features(x_spatial)  # torch.Size([64, 16, 42, 6])
        x_spectral = x_spatial + x_spectral
        x_spectral= self.relu (x_spectral)
        return x



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)




class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)    缩放操作
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads  #获得输入x的维度和多头注意力头的个数，x: [batch_size,patch_num,dim]
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3,.chunk功能：将数据拆分为特定数量的块
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions，对qkv的维度进行调整。 #map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  ## 爱因斯坦求和约定（einsum）,隐含语义：当bhi,bhj固定时，得到两个长度为d的向量，元素相乘，并对d长度求和
        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block #out torch.Size([64, 8, 5, 8])
        out = self.nn1(out)
        out = self.do1(out)
        return out


class cross_Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)    缩放操作
        self.to_qk= nn.Linear(dim, dim * 2, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x1, x2):
        b, n, _, h = *x1.shape, self.heads  #获得输入x的维度和多头注意力头的个数，x: [batch_size,patch_num,dim]
        qk = self.to_qk(x1).chunk(2, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3,.chunk功能：将数据拆分为特定数量的块
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qk)  # split into multi head attentions，对qkv的维度进行调整。 #map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
        v = rearrange(x2, 'b n (h d) -> b h n d', h=h)
        # print(q.shape)
        # print(v.shape)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  ## 爱因斯坦求和约定（einsum）,隐含语义：当bhi,bhj固定时，得到两个长度为d的向量，元素相乘，并对d长度求和
        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block #out torch.Size([64, 8, 5, 8])
        out = self.nn1(out)
        out = self.do1(out)
        return out




class feature_all(nn.Module):  # nn.Module是在pytorch使用非常广泛的类，搭建网络基本都需要用到这个。
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv3d_features1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Dropout(0.1)
        )  # nn.Sequential这是一个有顺序的容器,将特定神经网络模块按照在传入构造器的顺序依次被添加到计算图中执行。

        self.conv3d_features2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=12, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Dropout(0.1)
        )  # nn.Sequential这是一个有顺序的容器,将特定神经网络模块按照在传入构造器的顺序依次被添加到计算图中执行。
        
        self.globel_pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool3d((8,4,4)),
            nn.BatchNorm3d(12),
            nn.ReLU(),
           )

    def forward(self, x):
        x = self.conv3d_features1(x) 
        x = self.conv3d_features2(x)
        x = self.globel_pool_layer(x)
        return x


class MyGaussianBlur():
    # 初始化
    def __init__(self, patch=3, sigema=1.5, spe_dim=15):
        self.patch = patch
        self.sigema = sigema
        self.spe_dim = spe_dim

    # 高斯的计算公式
    def calc(self, x, y):
        res1 = 1 / (2 * math.pi * self.sigema * self.sigema)
        res2 = math.exp(-(x * x + y * y) / (2 * self.sigema * self.sigema))
        return res1 * res2

    # 滤波模板
    def template(self):
        sideLength = self.patch
        radius = (self.patch - 1) / 2
        result = np.zeros((sideLength, sideLength))
        for i in range(0, sideLength):
            for j in range(0, sideLength):
                result[i, j] = self.calc(i - radius, j - radius)
        all = result.sum()
        return result / all

    def guass_weight(self, template):
        kernel = np.array(template)
        kernel2 = torch.FloatTensor(kernel).expand(1, self.spe_dim, self.patch, self.patch)
        # weight = torch.nn.Parameter(data=kernel2, requires_grad=False)
        return kernel2


class mynet(nn.Module):
    def __init__(self,pca_components=15,patch_size=13,sigema=1,num_classes=16,  dim=96,  heads=16, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(mynet, self).__init__()

        self.GBlur = MyGaussianBlur(patch=patch_size, sigema=sigema, spe_dim=pca_components)  # 声明高斯模糊类
        self.temp = self.GBlur.template()  # 得到滤波模版
        self.gs_weight = self.GBlur.guass_weight(self.temp)
        self.pos_embedding1 = nn.Parameter(self.gs_weight)

        #torch.nn.init.normal_(self.pos_embedding1, std=.02)   #服从~ N(mean,std)

        self.feature_all= feature_all(in_channels=1)
        self.ss1 = SS()
        self.ss2 = SS()
        self.ss3 = SS()
        self.ss4 = SS()
        # self.pos_embedding1 = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        # torch.nn.init.normal_(self.pos_embedding1, std=.02)   #服从~ N(mean,std)
        self.cls_token = nn.Parameter(torch.empty(1, 1, dim))
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.dropout1 = nn.Dropout(emb_dropout)
        self.attention = Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout)))
        self.mlp = Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
        #self.to_cls_token = nn.Identity() #不区分参数的占位符标识运算符,输入是啥，直接给输出，不做任何的改变
        #self.attention_out = Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout)))
        # self.pos_embedding2 = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        # torch.nn.init.normal_(self.pos_embedding2, std=.02)   #服从~ N(mean,std)
        self.globel_pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d((1)),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
           )

        self.cross_attention =   cross_Attention(dim, heads=heads, dropout=dropout)
        self.norm = nn.BatchNorm1d(1)
        self.dropout2 = nn.Dropout(emb_dropout)

        self.nn1 = nn.Linear(dim, num_classes)

    def forward(self, x):
        
        x += self.pos_embedding1

        x = self.feature_all(x)
        x2 = self.ss1(x)
        x2 = self.ss2(x2)
        x2_1 = rearrange(x2, 'b c r h w -> b   (h w ) (c r)')
        x1 = rearrange(x, 'b c r h w -> b   (h w ) (c r)')
        cls_tokens = self.cls_token.expand(x1.shape[0], -1, -1)  #[64,1,64]其将单个维度扩大成更大维度，返回一个新的tensor,在expand中的-1表示取当前所在维度的尺寸，也就是表示当前维度不变。
        x1 = torch.cat((cls_tokens, x1), dim=1)   #[64,5,64]
        #x1 += self.pos_embedding1
        x1 = self.dropout1(x1)
        x1 = self.attention(x1)  # go to attention
        cls_tokens_ex1 = x1[:, 0]
        x_remain1 = x1[:,1:,:] 
        x1=x_remain1+x2_1
        x1 = torch.cat((cls_tokens_ex1.unsqueeze(1), x1), dim=1)
        x1 = self.mlp(x1)  #x1 torch.Size([64, 4, 64])
        cls_tokens_ex2 = x1[:, 0]
        #print(cls_tokens_ex2.shape)
        x_remain2 = x1[:,1:,:] 
        #print('remain',x_remain1)
        x2_2 = self.ss3(x2)
        x2_2 = self.ss4(x2_2)
        x2_2 = rearrange(x2_2, 'b c r h w -> b  (h w) (c r)')
        x1=x_remain2+x2_2

        x_remainout = rearrange(x1, 'b h w -> b  w h')
        x_remainout=  self.globel_pool_layer(x_remainout)
        x_remainout = rearrange(x_remainout, 'b h w -> b  w h')
        #print(x_remainout.shape)
        #xout = torch.cat((cls_tokens_ex2.unsqueeze(1), x_remainout), dim=1)
        #print((cls_tokens_ex2.unsqueeze(1)).shape)
        #xout += self.pos_embedding2
        #xout = self.dropout2(xout)
        xout = self.cross_attention(x_remainout,cls_tokens_ex2.unsqueeze(1))
        xout = self.norm(xout)
        xout = self.dropout2(xout)
        #xout = xout[:, 0]
        xout = rearrange(xout, 'b h w -> b  (w h)')
        x = self.nn1(xout)
        return x


if __name__ == '__main__':
    model = mynet()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 13, 13)
    y = model(input).apply(weights_init)
    print(y.size())



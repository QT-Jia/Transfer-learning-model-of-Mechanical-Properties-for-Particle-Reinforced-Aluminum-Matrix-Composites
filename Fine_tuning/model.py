import copy
import math
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0")
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, encode_layer, N):
        super(Encoder, self).__init__()
        self.layer_list = clones(encode_layer, N)

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

class SublayeResConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayeResConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x, sub_layer):
        return self.norm(x + self.Dropout(sub_layer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, multi_attention, feed_forward, dropout, init):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        if init:
            multi_attention.apply(weights_init)
        self.attention = multi_attention
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.sublayer = clones(SublayeResConnection(self.d_model, self.dropout), 2)

    def forward(self, x):
        x = self.sublayer[0](x, self.attention)
        x = self.sublayer[1](x, self.feed_forward)
        return x


def SelfAttention(q, k, v, scale_factor=None):
    if scale_factor is None:
        scale_factor = 1 / math.sqrt(k.size(-1))

    k_t = torch.transpose(k, dim0=-2, dim1=-1)
    q_k_t = torch.matmul(q, k_t)

    mask = (q_k_t != 0).float()
    q_k_t = q_k_t.masked_fill(mask == 0, -1e9)

    soft_value = nn.Softmax(dim=-1)(q_k_t / scale_factor)
    z = torch.matmul(soft_value, v)
    return z, soft_value


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_forward, head, dropout):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_forward = d_forward
        self.head = head
        self.dropout = dropout
        self.atten_score = None
        self.d_k = self.d_forward // self.head
        self.LinearList0 = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.head)])
        self.LinearList1 = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.head)])
        self.LinearList2 = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.head)])
        self.Linear = nn.Linear(self.d_forward, self.d_model)

    def forward(self, x):
        device = x.device
        n_batch = x.size(0)
        q = torch.stack([linear(x) for linear in self.LinearList0], dim=1)
        k = torch.stack([linear(x) for linear in self.LinearList1], dim=1)
        v = torch.stack([linear(x) for linear in self.LinearList2], dim=1)

        z, self.atten_score = SelfAttention(q, k, v, scale_factor=self.d_k ** -0.5)
        z = z.transpose(1, 2).reshape(n_batch, -1, self.head * self.d_k)
        return self.Linear(z)


class AtomFeedForward(nn.Module):
    def __init__(self, d_model, d_forward, dropout=0):
        super(AtomFeedForward, self).__init__()
        self.d_model = d_model
        self.Linear1 = nn.Linear(d_model, d_forward)
        self.Linear2 = nn.Linear(d_forward, d_model)
        self.ReLu = nn.LeakyReLU()
        self.Dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        r = self.Linear1(x)
        r = self.ReLu(r)
        r = self.Dropout(r)
        r = self.Linear2(r)
        return r


class LpBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(LpBlock, self).__init__()
        self.Linear = nn.Linear(input_size, output_size)
        self.BatchNormal = nn.BatchNorm1d(output_size)
        self.ReLu = nn.ReLU()

    def forward(self, x):
        x = self.Linear(x)
        x = self.BatchNormal(x)
        x = self.ReLu(x)
        return x


def weights_init(m, key='xavier_uniform'):
    if hasattr(m, 'weight'):
        if key == 'xavier_uniform':
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        elif key == 'kaiming_uniform':
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class MaskedMSELoss(nn.Module):
    def __init__(self, weights=None):
        super(MaskedMSELoss, self).__init__()
        self.weights = weights if weights is not None else torch.tensor([1.0, 1.0, 1.0])

    def forward(self, pred, target):
        mask = (target != 0).float()
        weighted_loss = ((pred - target) ** 2) * mask * self.weights.to(pred.device).view(1, -1)
        return weighted_loss.sum() / mask.sum()


class Pre_AlPP(nn.Module):
    def __init__(self, d_model, d_forward, head, dropout, attention_num, d_forward1, scale_factor):
        super(Pre_AlPP, self).__init__()
        self.d_model = d_model
        self.d_forward = d_forward
        self.head = head
        self.dropout = dropout
        self.attention_num = attention_num
        self.d_forward1 = d_forward1
        self.scale_factor = scale_factor
        c = copy.deepcopy

        # 使用register_buffer注册element张量
        self.register_buffer('element', torch.LongTensor(np.array(list(range(17)))))

        self.Embeddings = nn.Embedding(17, d_model)
        self.encoder_layer = EncoderLayer(self.d_model,
                                          c(MultiHeadAttention(self.d_model, self.d_forward, self.head, self.dropout)),
                                          c(AtomFeedForward(self.d_model, self.d_forward1)),
                                          self.dropout, True)
        self.TransFormerEncoder = Encoder(self.encoder_layer, self.attention_num)
        self.Dropout = nn.Dropout(p=0.1)
        self.process_layer = nn.Sequential(
            nn.Linear(4, 26),
            nn.ReLU(),
            nn.BatchNorm1d(26)
        )
        self.lp_layer = nn.ModuleList([
            LpBlock(128, 256),
            LpBlock(256, 512)
        ])
        self.Linear = nn.Linear(512, 4)

    def forward(self, x):
        device = next(self.parameters()).device

        if isinstance(x, list):
            x = [tensor.to(device) for tensor in x]

        atom_embedding = x[0][:, 0, :, :] * (x[0][:, 1, :, :] + self.Embeddings(self.element) * self.scale_factor)
        r = atom_embedding
        for layer in self.TransFormerEncoder.layer_list:
            r = layer.sublayer[0](r, layer.attention)
            self.attention_weights.append(layer.attention.atten_score)
            r = layer.sublayer[1](r, layer.feed_forward)

        r = r.flatten(1)
        r = self.Dropout(r)

        c = self.process_layer(x[-1])
        r = torch.cat([r, c], dim=-1)

        for layer in self.lp_layer:
            r = layer(r)
        return self.Linear(r)
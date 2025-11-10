import torch
import torch.nn as nn
import numpy as np
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import (
    Pre_AlPP, 
    EncoderLayer, 
    MultiHeadAttention, 
    AtomFeedForward, 
    Encoder,
    LpBlock,
)

class AlC_MP(Pre_AlPP):
    def __init__(self, pretrained_model_path, d_model=6, d_forward=16, head=4, dropout=0.1, attention_num=2, scale_factor= 0.1,
                 tune_encoder: bool = False, freeze_pretrained_encoder: bool = True):
        super(AlC_MP, self).__init__(
            d_model=d_model,
            d_forward=d_forward,
            head=head,
            dropout=dropout,
            attention_num = attention_num,
            scale_factor=scale_factor
        )

        self.tune_encoder = bool(tune_encoder)
        self.freeze_pretrained_encoder = bool(freeze_pretrained_encoder)
        self.process_layer = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        pretrained_dict = torch.load(pretrained_model_path, weights_only=False)
        
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and (
                              'TransFormerEncoder' in k or 
                              'process_layer' in k or 
                              'lp_layer' in k)}

        if 'process_layer.0.weight' in pretrained_dict:
            pretrained_weight = pretrained_dict['process_layer.0.weight']
            current_weight = model_dict['process_layer.0.weight']
            with torch.no_grad():
                current_weight[:, 3:7] = pretrained_weight
                model_dict['process_layer.0.weight'] = current_weight

        if self.freeze_pretrained_encoder:
            for layer in self.TransFormerEncoder.layer_list:
                for sublayer in layer.sublayer:
                    for param in sublayer.parameters():
                        param.requires_grad = False

        if 'lp_layer.0.Linear.weight' in pretrained_dict:
            lp_dict = {k: v for k, v in pretrained_dict.items() if 'lp_layer' in k}
            self.lp_layer.load_state_dict(lp_dict)  # lp_layer可训练

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        self.register_buffer('particle', torch.LongTensor(np.array(list(range(6)))))
        self.register_buffer('element', torch.LongTensor(np.array(list(range(17)))))

        self.P_Embeddings = nn.Embedding(6, 5)
        self.particle_encoder_layer = EncoderLayer(
            d_model=5,
            multi_attention=MultiHeadAttention(5,self.d_forward, self.head, self.dropout),
            feed_forward=AtomFeedForward(5, 64),dropout=self.dropout,init=True)
        self.particle_encoder = Encoder(self.particle_encoder_layer, self.attention_num)
        self.fusion_layer = nn.Sequential(
            nn.Linear(196, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.lp_layer = nn.ModuleList([
            LpBlock(256, 512),
            LpBlock(512, 256)
        ])
        self.output_layer = nn.Linear(256, 2) 


    def forward(self, x):
        device = next(self.parameters()).device

        if isinstance(x, list):
            x = [tensor.to(device) for tensor in x]
        atom_embedding = x[0][:, 0, :, :] * (
            x[0][:, 1, :, :] + 
            self.Embeddings(self.element) * 
            self.scale_factor
        )

        with torch.set_grad_enabled(self.tune_encoder):
            r = atom_embedding
            for layer in self.TransFormerEncoder.layer_list:
                r = layer.sublayer[0](r, layer.attention)
                self.attention_weights.append(layer.attention.atten_score)
                self.atom_attention_weights.append(layer.attention.atten_score)
                r = layer.sublayer[1](r, layer.feed_forward)

            r = r.flatten(1)

        with torch.set_grad_enabled(self.tune_encoder):
            c = self.process_layer(x[1])

        particle_data = x[-1].to(device)
        p1 = particle_data[:, 0, :, :]
        p2 = particle_data[:, 1, :, :]
        p = p1 * (p2 + self.P_Embeddings(self.particle) * self.scale_factor)

        p = p.flatten(1)
        p = self.Dropout(p)
        combined_features = torch.cat([r, c, p], dim=1)
        combined_features = self.fusion_layer(combined_features)
        
        for layer in self.lp_layer:
            combined_features = layer(combined_features)
            
        return self.output_layer(combined_features)


class MaskedMSELoss(nn.Module):
    def __init__(self, weight_ratio, weights=None):
        super(MaskedMSELoss, self).__init__()
        self.weight_ratio = weight_ratio
        self.weights = weights if weights is not None else torch.tensor([1.0, weight_ratio])

    def forward(self, pred, target):
        mask = (target != 0).float()
        weighted_loss = ((pred - target) ** 2) * mask * self.weights.to(pred.device).view(1, -1)
        return weighted_loss.sum() / mask.sum()
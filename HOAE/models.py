import numpy as np
import torch
from torch import nn
from layers import GC_HAN, HGAT, GAT
import torch.nn.functional as F


class GC_HAN_AC(nn.Module):
    def __init__(self,
                 num_metapaths,
                 in_dims,
                 hidden_dim,
                 out_dim,
                 num_classes,
                 num_heads,
                 graph,
                 meta_graph,
                 GCN_hidden_dim,
                 device,
                 ac_drop,
                 ac_layers,
                 agg,
                 dropout_rate=0.5,
                 cuda=False):
        super(GC_HAN_AC, self).__init__()
        self.g = graph
        self.device = device
        self.hidden_dim = hidden_dim
        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(m, hidden_dim, bias=True) for m in in_dims])

        # feature dropout after attribute completion
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(ac_drop)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        num_layers = ac_layers
        heads = [num_heads] * num_layers + [1]
        self.HGAT = HGAT(hidden_dim, hidden_dim, hidden_dim, num_classes, dropout_rate, dropout_rate, 0.05, num_layers,
                         heads, self.device)
        self.GAT = GAT(hidden_dim, hidden_dim, dropout_rate, dropout_rate, 1, device)
        # 下游模型
        self.layer1 = GC_HAN(num_metapaths, hidden_dim, out_dim, num_heads,
                             graph, meta_graph, GCN_hidden_dim, agg, attn_drop=dropout_rate)
        self.Leaner_Classification = nn.Linear(out_dim * num_heads, num_classes, bias=True)
        # self.Leaner_Classification = nn.Linear(256, num_classes, bias=True)

    def forward(self, inputs1):
        adj, feat_list, type_mask = inputs1
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=feat_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(feat_list[i])
        feat_src = transformed_features
        if transformed_features.shape[0] != 11616:
            _, transformed_features = self.HGAT(self.g, feat_src)
        else:
            transformed_features = self.GAT(self.g, feat_src)
            transformed_features = torch.squeeze(transformed_features, 1)
        transformed_features[0:4019] = feat_src[0:4019]

        transformed_features = self.feat_drop(transformed_features)
        # hidden layers
        h = self.layer1(transformed_features)
        logits = self.Leaner_Classification(h)

        return logits, h

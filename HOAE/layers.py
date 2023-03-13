import math
import numpy as np
import torch
from torch import nn
from dgl.nn.pytorch import GraphConv,GATConv
# The program of valid Transformer
# from gat_transformer import GATConv
import torch.nn.functional as F

class GC_HAN(nn.Module):
    def __init__(self,
                 num_metapaths,
                 in_dim,
                 out_dim,
                 num_heads,
                 graph,
                 meta_graph,
                 GCN_hidden_dim,
                 agg,
                 attn_drop=0.5):

        super(GC_HAN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.agg = agg
        self.graph = graph
        self.meta_graph = meta_graph
        # GCN
        self.GAT_feat_drop = attn_drop
        self.GCN = GCN(in_dim, GCN_hidden_dim)
        self.GAT = GATConv(GCN_hidden_dim, GCN_hidden_dim, 1, self.GAT_feat_drop, attn_drop, activation=F.elu)
        self.num_metapaths = num_metapaths
        self.dropout = nn.Dropout(attn_drop)
        self.gat_layers = nn.ModuleList()
        for i in range(num_metapaths):  # meta-path Layers
            self.gat_layers.append(GATConv(GCN_hidden_dim, out_dim, num_heads,
                                           self.GAT_feat_drop, attn_drop, activation=F.elu))

        self.semantic_attention = SemanticAttention(in_size=out_dim * num_heads)

    def forward(self, features):
        if self.agg == "mean":
            h = self.GCN(features, self.graph)
        elif self.agg == "att":
            h = self.GAT(self.graph, features)
        else:
            raise IOError("Aggregator input error!")
        h = self.dropout(h)
        if features.shape[0] == 11246:
            h = h[0:4019].flatten(1)
        elif features.shape[0] == 11616:
            h = h[0:4278].flatten(1)
        elif features.shape[0] == 3913:
            h = h[0:2614]
        elif features.shape[0] == 26128:
            h = h[0:4057]
        else:
            raise Exception("dataset exception!")
        embeddings = []
        for i, g in enumerate(self.meta_graph):
            embeddings.append(self.gat_layers[i](g, h).flatten(1))
        embeddings = torch.stack(embeddings, dim=1)
        h = self.semantic_attention(embeddings)
        return h


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.activation = F.elu
        self.gcn_layers = GraphConv(input_dim, output_dim, activation=self.activation)

    def forward(self, features, graph):
        h = self.gcn_layers(graph, features)
        return h


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(SemanticAttention, self).__init__()
        # input:[Node, metapath, in_size]; output:[None, metapath, 1]
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class HGAT(nn.Module):
    def __init__(self, in_dims, nhid, l_hid, nclass, feat_drop, attn_drop, negative_slope, num_layers, nheads, device):
        """Dense version of GAT."""
        super(HGAT, self).__init__()
        self.device = device
        self.l_hid = l_hid
        self.activation = F.elu

        self.gat_layers = nn.ModuleList()
        self.num_layers = num_layers
        # input layer
        self.gat_layers.append(GATConv(
            l_hid, nhid, nheads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))

        # hidden layer
        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(
                nhid * nheads[l-1], nhid, nheads[l], feat_drop, attn_drop, negative_slope, self.activation
            ))

        # output layer
        self.gat_layers.append(GATConv(
            nhid * nheads[-2], nhid, nheads[-1], feat_drop, attn_drop, negative_slope, None))

        self.Leaner_Classification = nn.Linear(nhid * nheads[-1], nclass, bias=True)

    def forward(self, g, h):
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        h = self.gat_layers[-1](g, h).mean(1)
        logits = self.Leaner_Classification(h)
        return logits, h


class GAT(nn.Module):
    def __init__(self, in_dims, nhid, feat_drop, attn_drop, nheads, device):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.device = device
        self.activation = F.elu
        self.gat_layers = GATConv(in_dims, nhid, nheads, feat_drop, attn_drop, activation=F.elu)

    def forward(self, g, h):
        # 调用GAT
        h = self.gat_layers(g, h)
        # logits = self.Leaner_Classification(h)
        return h

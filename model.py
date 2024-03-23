import dgl
import torch
import torch.nn as nn
import dgl.function as fn
from global_args import device

class DiGAMN_Model(nn.Module):
    """
    Graph Attention Multi-layer Network (GAMN) implementation.
    """
    def __init__(self, params):
        super(DiGAMN_Model, self).__init__()
        self.n_hid = params.n_hid  # Number of hidden units
        self.n_layers = params.MGAT_layer  # Number of MGAT layers
        self.mem_size = params.Memory_size  # Size of the memory encoding
        self.GAT_layer = params.GAT_layer  # Number of GAT sub-layers

        # Embedding matrix initialization
        self.emb = nn.Parameter(torch.empty(params.num_nodes, self.n_hid)).to(device)
        # Normalization layer
        self.norm = nn.LayerNorm((self.n_layers + 1) * self.n_hid).to(device)

        # Stacking MGATLayer modules
        self.gmlayers = nn.ModuleList([GAMNLayer(self.n_hid, self.n_hid, self.mem_size, params.num_rels, self.GAT_layer,
                                               True, nn.LeakyReLU(0.2, inplace=True), params.dropout).to(device)
                                     for _ in range(self.n_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes or resets the parameters of the model."""
        nn.init.normal_(self.emb)

    def predict(self, user, item):
        """Predicts the score between user and item embeddings."""
        return torch.einsum('bc, bc -> b', user, item).to(device) / user.shape[1]

    def forward(self, graph, x_encode):
        """
        Forward pass through the model.
        Args:
            graph (DGLGraph): The graph.
            x_encode (Tensor): Encoded features tensor.
        """
        x = self.emb.to(device)
        all_emb = [x]
        for layer in self.gmlayers:
            x = layer(graph, x)
            all_emb.append(x)
        x = torch.cat(all_emb, dim=1)
        x = self.norm(x)
        return x


class BioMemorylayer(nn.Module):
    """
    A module for bio memory encoding in GAMN.
    """
    def __init__(self, in_feats, out_feats, mem_size):
        super(BioMemorylayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.mem_size = mem_size
        self.linear_coef = nn.Linear(in_feats, mem_size, bias=True).to(device)
        self.linear_coef2 = nn.Linear(in_feats, mem_size, bias=False).to(device)
        self.act = nn.LeakyReLU(0.2, inplace=True).to(device)
        self.linear_w = nn.Linear(mem_size, out_feats * in_feats, bias=False).to(device)

    def forward(self, h_dst, h_src):
        """Forward pass for bio memory encoding."""
        coef = self.act(self.linear_coef(h_dst))
        w2 = self.linear_coef2(h_src).to(device)
        w = self.linear_w(coef).view(-1, self.out_feats, self.in_feats).to(device)
        res = torch.einsum('boi, bi -> bo', w, w2).to(device)
        return res


class GAMNLayer(nn.Module):
    """
    Multi-head Graph Attention Layer with Memory Encoding.
    """
    def __init__(self, in_feats, out_feats, mem_size, num_rels, gat_layers, bias=True, activation=None, dropout=0.0, layer_norm=False):
        super(GAMNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.mem_size = mem_size
        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.edge_BML = nn.ModuleList([
            BioMemorylayer(in_feats, out_feats, mem_size).to(device)
            for _ in range(num_rels)
        ])

        self.attentions = nn.ModuleList([
            dgl.nn.GATConv(in_feats, out_feats, num_heads=4).to(device)
            for _ in range(gat_layers)
        ])

        if bias:
            self.h_bias = nn.Parameter(torch.empty(out_feats)).to(device)
            nn.init.zeros_(self.h_bias)
        else:
            self.h_bias = None

        if layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feats).to(device)
        else:
            self.layer_norm_weight =None

    def forward(self, g, feat):
        """Forward pass for MGATLayer."""
        with g.local_scope():
            g.ndata['h'] = feat
            for attention in self.attentions:
                g.ndata['h'] = torch.sum(attention(g, feat), dim=1).to(device)

            g.update_all(self.message_func, fn.mean(msg='m', out='h'))
            node_rep = g.ndata['h']

            if self.layer_norm_weight is not None:
                node_rep = self.layer_norm_weight(node_rep)

            if self.bias is not None:
                node_rep += self.h_bias

            if self.activation is not None:
                node_rep = self.activation(node_rep)

            return self.dropout(node_rep)

    def message_func(self, edges):
        """Custom message function for edge updates."""
        msg = torch.empty((edges.src['h'].size(0), self.out_feats), device=edges.src['h'].device)
        for etype in range(self.num_rels):
            loc = edges.data['type'] == etype
            if loc.sum() == 0:
                continue
            src = edges.src['h'][loc]
            dst = edges.dst['h'][loc]
            sub_msg = self.edge_BML[etype](dst, src)
            msg[loc] = sub_msg
        return {'m': msg}


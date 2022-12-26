import torch
import torch.nn as nn
import math
import numpy as np


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim, tanh_clipping):
        super(SingleHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.tanh_clipping = tanh_clipping

        self.input_dim = self.embedding_dim
        self.embedding_dim = self.embedding_dim
        self.value_dim = self.embedding_dim
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp

        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim

        h_flat = h.reshape(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.reshape(-1, input_dim)  # (batch_size*n_query)*input_dim

        shape_k = (batch_size, target_size, -1)
        shape_q = (batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # batch_size*targets_size*key_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))  # batch_size*n_query*targets_size
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            mask = mask.view(batch_size, 1, target_size).expand_as(U)  # copy for n_heads times
            U[mask.bool()] = -1e8
        attention = torch.log_softmax(U, dim=-1)  # batch_size*n_query*targets_size

        out = attention

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim=None, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim

        self.query_dim = self.embedding_dim  # step_context_size
        self.input_dim = self.embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.query_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp
        assert q.size(0) == batch_size
        # assert q.size(2) == input_dim
        assert input_dim == self.input_dim

        h_flat = h.contiguous().view(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.contiguous().view(-1, self.query_dim)  # (batch_size*n_query)*input_dim
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)
        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(h_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim
        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size
        if mask is not None:
            mask = mask.view(1, batch_size, 1, target_size).expand_as(U)  # copy for n_heads times
            U[mask.bool()] = -np.inf

        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            attnc = attention.clone()
            attnc[mask.bool()] = 0
            attention = attnc

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(batch_size, n_query, self.embedding_dim)

        return out  # batch_size*n_query*embedding_dim


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, inputs):
        return inputs + self.module(inputs)


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class AttentionLayer(nn.Module):
    # For not self attention
    def __init__(self, embedding_dim):
        super(AttentionLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, q, h, mask=None):
        h0 = q
        h = self.multiHeadAttention(q=q, h=h,mask=mask)
        h = h+h0
        h=self.normalization1(h)
        h1=h
        h = self.feedForward(h)
        h2 = h+h1
        h=self.normalization2(h2)
        return h


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.embedding_dim = cfg["model_config"]["embedding_dim"]
        self.tanh_clipping = cfg["model_config"]["tanh_clipping"]

        self.target_MHA = AttentionLayer(embedding_dim=self.embedding_dim)
        self.agent_MHA = AttentionLayer(embedding_dim=self.embedding_dim)
        self.MHA = AttentionLayer(embedding_dim=self.embedding_dim)
        self.SHA = SingleHeadAttention(embedding_dim=self.embedding_dim, tanh_clipping=self.tanh_clipping)

    def forward(self, current_state, target_feature, agent_feature, mask, decode_type='sampling'):
        # target_embedding = target_feature  # (batch_size,target_size,embed_size) (batch_size,embed_size)
        # batch_size, target_size, embedding_size = target_embedding.size()
        h_c = current_state
        target_h_c = self.agent_MHA(q=target_feature, h=agent_feature) # task-agent encoder
        h_c_prime = self.target_MHA(q=h_c, h=agent_feature) # current state embedding
        h_c_prime = self.MHA(q=h_c_prime, h=target_h_c, mask=mask) # final candidate embedding
        log_prob = self.SHA(q=h_c_prime, h=target_h_c, mask=mask).squeeze(1)  # [batch_size,target_size]

        if decode_type == 'sampling':
            next_target_index = torch.multinomial(log_prob.exp(), 1).long().squeeze(1)
        else:
            next_target_index = torch.argmax(log_prob, dim=1).long()
            if next_target_index.item()==0:
                next_target_index = torch.multinomial(log_prob.exp(), 1).long().squeeze(1)

        return next_target_index, log_prob # return prob

    # def get_score_function(self, _log_p, pi):
    #     """	args:
    #         _log_p: (batch, city_t, city_t)
    #         pi: (batch, city_t), predicted tour
    #         return: (batch) sum of the log probability of the chosen targets
    #     """
    #     log_p = torch.gather(input=_log_p, dim=2, index=pi[:, :, None])
    #     return torch.sum(log_p.squeeze(-1), 1)

    # def sum_distance(self, inputs, route):
    #     d = torch.gather(input=inputs, dim=1, index=route[:, :, None].repeat(1, 1, 2))
    #     return (torch.sum((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2), dim=1)
    #             + (d[:, 0] - d[:, -1]).norm(p=2, dim=1))  # distance from last node to first selected node)



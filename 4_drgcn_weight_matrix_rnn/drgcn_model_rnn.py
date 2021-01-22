import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv

class DRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases, num_rgcn_hidden_layers=1, use_self_loop=False,
                num_dkrl_hidden_layers=1, rnn_hidden_size=100, wordNum=0,
                dropout=0):
        super(DRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_rgcn_hidden_layers = num_rgcn_hidden_layers
        self.use_self_loop = use_self_loop
        
        self.num_dkrl_hidden_layers = num_dkrl_hidden_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.wordNum = wordNum

        self.dropout = dropout

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.rgcnLayers = nn.ModuleList()
        # i2h, rgcn embedding
        i2h = self.build_input_layer()     # Entity to embedding
        if i2h is not None:
            self.rgcnLayers.append(i2h)
        # h2h
        for idx in range(self.num_rgcn_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.rgcnLayers.append(h2h)

        self.dkrlLayers = nn.ModuleList()       # dkrl CNN word2vec part
        # d2h, description embedding
        d2h = self.build_desc_input_layer()     # Entity descriptions(word) to embedding
        if d2h is not None:
            self.dkrlLayers.append(d2h)
        # for idx in range(self.num_dkrl_hidden_layers):
        #     dh2h = self.build_desc_hidden_layer(idx)
        #     self.dkrlLayers.append(dh2h)
        dh2h = self.build_desc_hidden_layer()
        if dh2h is not None:
            self.dkrlLayers.append(dh2h)
        
        # We have no output layer: the last hidden layer is the output layer.
        #   The last output layer will output a new entity embeddings by DGL -> RelGraphConv().
        # # h2o
        # h2o = self.build_output_layer()
        # if h2o is not None:
        #     self.layers.append(h2o)

    # rgcn embedding
    def build_input_layer(self):
        entityEmbedding = EmbeddingLayer(num_nodes=self.num_nodes, h_dim=self.h_dim)
        return entityEmbedding
    
    # description embedding
    def build_desc_input_layer(self):
        descEmbedding = DescEmbeddingLayer(wordNum=self.wordNum, h_dim=self.h_dim)
        return descEmbedding

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_rgcn_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                self.num_bases, activation=act, self_loop=self.use_self_loop,
                dropout=self.dropout)
        
    def build_desc_hidden_layer(self):
        return Desc2VecRNN(embedding_size=self.h_dim, hidden_size=self.rnn_hidden_size, num_layers=self.num_dkrl_hidden_layers, dropout_rate=self.dropout, bidirectional=True)
    
    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm, s_e_d_w_embeddings):        # desch: from entity idx, use description -> word2vec -> cnn -> entity_cnn representation
        for rgcnLayer in self.rgcnLayers:
            h = rgcnLayer(g, h, r, norm)
        
        for dkrlLayer in self.dkrlLayers:
            s_e_d_w_embeddings = dkrlLayer(s_e_d_w_embeddings)
        
        return h, s_e_d_w_embeddings

class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        # entity embedding
        self.entityEmbedding = nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):       # parameters: graph, entity_id, relation_id, normalizer(maybe)
        return self.entityEmbedding(h.squeeze())

class DescEmbeddingLayer(nn.Module):
    def __init__(self, wordNum, h_dim):
        super(DescEmbeddingLayer, self).__init__()
        # word2vec
        self.wordEmbedding = nn.Embedding(wordNum+1, h_dim, padding_idx=0)     # 0: 0 padding, 1~: word embeddings
    
    def forward(self, s_e_d_w_embeddings):
        embeddings = torch.stack([self.wordEmbedding(words) for words in s_e_d_w_embeddings])
        # for words in s_e_d_w_embeddings:
        #     desc_embeddings = self.wordEmbedding(words)
        #     embeddings.append(desc_embeddings)
        # embeddings = torch.Tensor(embeddings)
        return embeddings

# Input: Word vectors in an entity's description
# Output: Entity embedding by its description
class Desc2VecRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, dropout_rate, bidirectional):
        super(Desc2VecRNN, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def attention_net(self, rnn_output, final_state):
        rnn_output = rnn_output[:, :, :self.hidden_size] + rnn_output[:, :, self.hidden_size:]
        rnn_output = rnn_output.permute(1, 0, 2)        # [batch, length, hidden_size]
        
        final_state = torch.sum(final_state, dim=0)     # [num_layers*num_directions, batch, hidden_size] -> [batch, hidden_size]

        attn_weights = torch.bmm(rnn_output, final_state.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(rnn_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state
    
    def forward(self, s_e_d_w_embeddings):
        batch_size = s_e_d_w_embeddings.shape[0]
        s_e_d_w_embeddings = s_e_d_w_embeddings.permute(1, 0, 2)

        output, (final_hidden_state, final_cell_state) = self.rnn(s_e_d_w_embeddings)

        # attention
        output = self.attention_net(output, final_hidden_state)

        output = self.dropout(output)
        output = self.fc(output)

        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv

class DRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_rgcn_hidden_layers=1, #num_dkrl_hidden_layers=1,
                 dropout=0, use_self_loop=False, use_cuda=False,
                 descDict={}, descWordNumDict={},
                 wordDict=set(), wordNum=0,
                 sampledDescWordNumMax=100):
        super(DRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_rgcn_hidden_layers = num_rgcn_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        #self.num_dkrl_hidden_layers = num_dkrl_hidden_layers
        self.descDict = descDict
        self.descWordNumDict = descWordNumDict
        self.wordDict = wordDict
        self.wordNum = wordNum
        self.sampledDescWordNumMax = sampledDescWordNumMax

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
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout)
        
    def build_desc_hidden_layer(self):
        # act = F.relu if idx < self.num_dkrl_hidden_layers - 1 else None
        return Desc2VecCNN(in_feat=self.h_dim, out_feat=self.h_dim,
                activation=nn.ReLU(), dropout=self.dropout, window_sizes=[1,2,3], sampledDescWordNumMax=self.sampledDescWordNumMax)

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
class Desc2VecCNN(nn.Module):
    def __init__(self, in_feat, out_feat, activation=None, dropout=0.0, window_sizes=[1,2,3], sampledDescWordNumMax=100):
        super(Desc2VecCNN, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.window_sizes = window_sizes
        self.s_e_d_w_maxNum = sampledDescWordNumMax

        # cnn model step
        # conv -> act -> maxpool -> fc
        self.convs = nn.ModuleList()
        self.maxpools = nn.ModuleList()
        for window_size in self.window_sizes:
            conv = nn.Conv1d(in_channels=self.in_feat, out_channels=self.out_feat, kernel_size=window_size)
            maxpool = nn.MaxPool1d(kernel_size=self.s_e_d_w_maxNum)

            self.convs.append(conv)
            self.maxpools.append(maxpool)
        self.fc = nn.Linear(in_features=self.in_feat*len(self.window_sizes), out_features=self.out_feat)

    
    # s: sampled, e: entity, d:description, w: word
    def forward(self, s_e_d_w_embeddings):
        s_e_d_w_embeddings = s_e_d_w_embeddings.permute(0, 2, 1)
        features_list = []
        for window_size in self.window_sizes:
            features = F.pad(s_e_d_w_embeddings, (0, window_size-1), mode='circular')
            features_list.append(features)
        
        all_features = []
        for i in range(len(self.window_sizes)):
            features = features_list[i]

            conv = self.convs[i]
            act = self.activation
            maxpool = self.maxpools[i]

            features = conv(features)
            features = act(features)
            features = maxpool(features)

            all_features.append(features)
        all_features = torch.cat(all_features, dim=1)
        all_features = all_features.flatten(start_dim=1)

        dropout = self.dropout
        fc = self.fc
        all_features = dropout(all_features)
        e_cnn_vec = fc(all_features)

        return e_cnn_vec

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseDRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_rgcn_hidden_layers=1, #num_dkrl_hidden_layers=1,
                 dropout=0, use_self_loop=False, use_cuda=False,
                 descDict={}, descWordNumDict={},
                 wordDict=set(), wordNum=0):
        super(BaseDRGCN, self).__init__()
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
        self.dkrlLayers.append(dh2h)
        
        # We have no output layer: the last hidden layer is the output layer.
        #   The last output layer will output a new entity embeddings by DGL -> RelGraphConv().
        # # h2o
        # h2o = self.build_output_layer()
        # if h2o is not None:
        #     self.layers.append(h2o)

    def build_input_layer(self):
        return None
    
    def build_desc_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_desc_hidden_layer(self):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm, s_e_d_w_embeddings, s_e_d_w_maxNum):        # desch: from entity idx, use description -> word2vec -> cnn -> entity_cnn representation
        for rgcnLayer in self.rgcnLayers:
            h = rgcnLayer(g, h, r, norm)
        
        for dkrlLayer in self.dkrlLayers:
            s_e_d_w_embeddings = dkrlLayer(s_e_d_w_embeddings, s_e_d_w_maxNum)
        
        return h, s_e_d_w_embeddings

# Input: Word vectors in an entity's description
# Output: Entity embedding by its description
class Desc2VecCNN(nn.Module):
    def __init__(self, in_feat, out_feat, activation=None, dropout=0.0, windows_size=[2,3,4]):
        super(Desc2VecCNN, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.windows_size = windows_size
    
    # s: sampled, e: entity, d:description, w: word
    def forward(self, s_e_d_w_embeddings, s_e_d_w_maxNum):
        s_e_d_w_embeddings = s_e_d_w_embeddings.permute(0, 2, 1)

        all_features = []
        for window_size in self.windows_size:
            features = F.pad(s_e_d_w_embeddings, (0, window_size-1), mode='circular')

            conv = nn.Conv1d(in_channels=self.in_feat, out_channels=self.out_feat, kernel_size=window_size)
            act = self.activation
            maxpool = nn.MaxPool1d(kernel_size=s_e_d_w_maxNum-window_size+1)

            features = conv(features)
            features = act(features)
            features = maxpool(features)

            all_features.append(features)
        
        all_features = torch.cat(all_features, dim=1)

        dropout = self.dropout
        fc = nn.Linear(in_features=self.in_feat*len(self.windows_size), out_features=self.out_feat)
        
        all_features = all_features.flatten(start_dim=1)
        all_features = dropout(all_features)
        e_cnn_vec = fc(all_features)

        return e_cnn_vec

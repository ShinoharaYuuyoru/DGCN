import argparse

import load_data
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
from drgcn_model import BaseDRGCN
from drgcn_model import Desc2VecCNN
import drgcn_utils

import matplotlib.pyplot as plt

class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        # entity embedding
        self.entityEmbedding = nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):       # parameters: graph, entity_id, relation_id, normalizer(maybe)
        return self.entityEmbedding(h.squeeze())
        # NEED PADDING

class DescEmbeddingLayer(nn.Module):
    def __init__(self, wordNum, h_dim):
        super(DescEmbeddingLayer, self).__init__()
        # word2vec
        self.wordEmbedding = nn.Embedding(wordNum+1, h_dim, padding_idx=0)     # 0: 0 padding, 1~: word embeddings
    
    def forward(self, s_e_d_w_embeddings, s_e_d_w_maxNum):
        embeddings = torch.stack([self.wordEmbedding(words) for words in s_e_d_w_embeddings])
        # for words in s_e_d_w_embeddings:
        #     desc_embeddings = self.wordEmbedding(words)
        #     embeddings.append(desc_embeddings)
        # embeddings = torch.Tensor(embeddings)
        return embeddings

class DRGCN(BaseDRGCN):
    # init is in BaseDRGCN

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
                activation=nn.ReLU(), dropout=self.dropout, windows_size=[2,3,4])
    
# As the scoring function, we will test these method at least:
#   1. DistMult
#   2. TransE

class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_rgcn_hidden_layers=1, #num_dkrl_hidden_layers=1,
                 dropout=0, use_cuda=False, reg_param=0, 
                 descDict={}, descWordNumDict={},
                 wordDict=dict(), wordNum=0):
        super(LinkPredict, self).__init__()

        # drgcn model
        self.drgcn = DRGCN(num_nodes=in_dim, h_dim=h_dim, out_dim=h_dim, num_rels=num_rels * 2, num_bases=num_bases,
                         num_rgcn_hidden_layers=num_rgcn_hidden_layers, #num_dkrl_hidden_layers=num_dkrl_hidden_layers,
                         dropout=dropout, use_self_loop=False, use_cuda=use_cuda,
                         descDict=descDict, descWordNumDict=descWordNumDict,
                         wordDict=wordDict, wordNum=wordNum)

        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))       # randomly init relation matrix
        nn.init.kaiming_uniform_(self.w_relation, nonlinearity='relu')      # xavier uniform performs not good with relu, so we use kaiming init.
        # nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        #   TBD
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm, s_e_d_w_embeddings, s_e_d_w_maxNum):
        return self.drgcn.forward(g, h, r, norm, s_e_d_w_embeddings, s_e_d_w_maxNum)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    # TBD
    def get_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']

def createWordDict(descDict):
    wordDict = dict()
    wordId = 1      # 0 for 0 padding in embedding.

    for entity in descDict:
        desc = descDict[entity]
        desc = desc.split()
        for word in desc:
            if word not in wordDict:
                wordDict[word] = wordId
                wordId += 1
    
    return wordDict, len(wordDict)

def getSampledDescWordList(node_id, descDict, descWordNumDict, wordDict):
    sampledDescWordNumMax = -1
    for entity in node_id:
        descWordNum = descWordNumDict[entity]
        if descWordNum > sampledDescWordNumMax:
            sampledDescWordNumMax = descWordNum
    sampledDescWordList = np.zeros((len(node_id), sampledDescWordNumMax), dtype=np.int32)
    for i, entity in enumerate(node_id):
        desc = descDict[entity].split()
        for j, word in enumerate(desc):
            wordId = wordDict[word]
            sampledDescWordList[i][j] = wordId
    
    return sampledDescWordList, sampledDescWordNumMax

def main(args):
    trainData, validData, testData, descDict, descWordNumDict, entityNum, relationNum = load_data.load_data(args.dataset)

    train_data = np.array(trainData)
    valid_data = np.array(validData)
    test_data = np.array(testData)

    num_nodes = entityNum
    num_rels = relationNum
    # Description statistic:
    #   FB15K-237: 87326 words
    #   WN18RR: 43856 words
    wordDict, wordNum = createWordDict(descDict)
    
    # # descWordNum statistic
    # maxDescWordNum = max(descWordNumDict.values())
    # sta = np.zeros(maxDescWordNum + 1, dtype=np.int32)
    # for val in descWordNumDict.values():
    #     sta[val] += 1
    # print(sta)

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create model
    model = LinkPredict(in_dim=num_nodes, h_dim=args.n_hidden, num_rels=num_rels, num_bases=args.n_bases,
                        num_rgcn_hidden_layers=args.n_rgcn_layers, #num_dkrl_hidden_layers=args.n_dkrl_layers,
                        dropout=args.dropout, use_cuda=use_cuda, reg_param=args.regularization,
                        descDict=descDict, descWordNumDict=descWordNumDict,
                        wordDict=wordDict, wordNum=wordNum)

    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    # build test graph
    test_graph, test_rel, test_norm = drgcn_utils.build_test_graph(
        num_nodes, num_rels, train_data)
    test_deg = test_graph.in_degrees(
                range(test_graph.number_of_nodes())).float().view(-1,1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

    if use_cuda:
        model.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = drgcn_utils.get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_state_file = 'model_state.pth'
    forward_time = []
    backward_time = []

    # training loop
    print("start training...")

    epoch = 0
    best_mrr = 0
    while True:
        model.train()
        epoch += 1

        # perform edge neighborhood sampling to generate training graph and data
        #   g: Half of sampled graph.
        #   node_id: unique entity ids.
        #   edge_type: relation ids.
        #   node_norm: normalized degrees of entities in graph g.
        #   data: (posSample, negSample)
        #   labels: (True*numPosSample, False*numNegSample)
        g, node_id, edge_type, node_norm, data, labels = \
            drgcn_utils.generate_sampled_graph_and_labels(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, degrees, args.negative_sample,
                args.edge_sampler)
        print("Done edge sampling")

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
            data, labels = data.cuda(), labels.cuda()
            g = g.to(args.gpu)
        
        # Get sampled entitiy descriptions and split the words.
        sampledDescWordList, sampledDescWordNumMax = getSampledDescWordList(node_id.flatten().tolist(), descDict, descWordNumDict, wordDict)
        sampledDescWordList = torch.from_numpy(sampledDescWordList).long()     # Convert to tensor

        # Calculate loss and backward
        #   TBD
        #   Problem: How to get the loss during rgcnEmbedding and dkrlEmbedding?
        t0 = time.time()
        # embed = model(g, node_id, edge_type, edge_norm)     # rgcn.forward(self, g, feat, etypes, norm=None)
        # loss = model.get_loss(g, embed, data, labels)
        rgcnEmbedding, dkrlEmbedding = model(g, node_id, edge_type, edge_norm, sampledDescWordList, sampledDescWordNumMax)     # rgcn.forward(self, g, feat, etypes, norm=None, desch)
        loss = model.get_loss(g, rgcnEmbedding, data, labels, dkrlEmbedding)        # TBD
        t1 = time.time()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

        optimizer.zero_grad()

        # validation
        if epoch % args.evaluate_every == 0:
            # perform validation on CPU because full graph is too large
            if use_cuda:
                model.cpu()
            model.eval()
            print("start eval")
            embed = model(test_graph, test_node_id, test_rel, test_norm)
            mrr = drgcn_utils.calc_mrr(embed, model.w_relation, torch.LongTensor(train_data),
                                 valid_data, test_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size,
                                 eval_p=args.eval_protocol)
            # save best model
            if mrr < best_mrr:
                if epoch >= args.n_epochs:
                    break
            else:
                best_mrr = mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           model_state_file)
            if use_cuda:
                model.cuda()

    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

    print("\nstart testing:")
    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    if use_cuda:
        model.cpu() # test on CPU
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    embed = model(test_graph, test_node_id, test_rel, test_norm)
    drgcn_utils.calc_mrr(embed, model.w_relation, torch.LongTensor(train_data), valid_data,
                   test_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size, eval_p=args.eval_protocol)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DGCN')

    # Required settings.
    #   When you use VSCode Python debugger to debug, check the launch.json configuration file.
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use. 'FB15K-237' or 'WN18RR'.")
    parser.add_argument("--gpu", type=int, required=True, help="GPU number(>=0) to use. Input '-1' to use CPU.")
    parser.add_argument("--eval_protocol", type=str, required=True, help="Type of evaluation probocol: 'raw' or 'filtered'.")

    # Optional settings.
    #   About the model
    #       Overall
    parser.add_argument("--n_hidden", type=int, default=100, help="Number of hidden units.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability.")
    parser.add_argument("--grad_norm", type=float, default=1.0, help="Norm to clip gradient to.")
    parser.add_argument("--regularization", type=float, default=0.01, help="Regularization weight.")
    #       RGCN
    parser.add_argument("--n_bases", type=int, default=100, help="Number of weight blocks for each relation.")
    parser.add_argument("--n_rgcn_layers", type=int, default=2, help="Number of RGCN layers / propagation rounds.")
    #       DKRL
    # parser.add_argument("--n_dkrl_layers", type=int, default=1, help="Number of DKRL layers / propagation rounds.")

    #   About training.
    parser.add_argument("--n_epochs", type=int, default=5000, help="Number of minimum training epochs.")
    parser.add_argument("--evaluate_every", type=int, default=500, help="Perform evaluation every n epochs.")
    parser.add_argument("--negative_sample", type=int, default=10, help="Number of negative samples per positive sample.")
    parser.add_argument("--graph_batch_size", type=int, default=30000, help="Number of edges to sample in each iteration.")
    parser.add_argument("--graph_split_size", type=float, default=0.5, help="Portion of edges used as positive sample.")
    parser.add_argument("--edge_sampler", type=str, default="uniform", help="Type of edge sampler: 'uniform' or 'neighbor'.")
    #   About evaluating.
    parser.add_argument("--eval_batch_size", type=int, default=500, help="Batch size when evaluating.")

    args = parser.parse_args()
    print(args)

    main(args)
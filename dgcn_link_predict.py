import argparse
import load_data
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
from model import BaseRGCN
import utils

class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())

class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout)

class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        # self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
        #                  num_hidden_layers, dropout, use_cuda)
        # BUG FIX: Should write as: use_cuda=use_cuda
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda=use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

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

def main(args):
    trainData, validData, testData, descDict, descWordNumDict, entityNum, relationNum = load_data.load_data(args.dataset)

    train_data = np.array(trainData)
    valid_data = np.array(validData)
    test_data = np.array(testData)
    num_nodes = entityNum
    num_rels = relationNum

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create model
    model = LinkPredict(num_nodes,
                        args.n_hidden,
                        num_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_cuda=use_cuda,
                        reg_param=args.regularization)

    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    # build test graph
    test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_rels, train_data)
    test_deg = test_graph.in_degrees(
                range(test_graph.number_of_nodes())).float().view(-1,1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

    if use_cuda:
        model.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

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
            utils.generate_sampled_graph_and_labels(
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

        t0 = time.time()
        embed = model(g, node_id, edge_type, edge_norm)     # rgcn.forward(self, g, feat, etypes, norm=None)
        loss = model.get_loss(g, embed, data, labels)
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
            mrr = utils.calc_mrr(embed, model.w_relation, torch.LongTensor(train_data),
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
    utils.calc_mrr(embed, model.w_relation, torch.LongTensor(train_data), valid_data,
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
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers / propagation rounds.")
    parser.add_argument("--n_hidden", type=int, default=500, help="Number of hidden units.")
    parser.add_argument("--n_bases", type=int, default=100, help="Number of weight blocks for each relation.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability.")
    parser.add_argument("--grad_norm", type=float, default=1.0, help="Norm to clip gradient to.")
    parser.add_argument("--regularization", type=float, default=0.01, help="Regularization weight.")
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
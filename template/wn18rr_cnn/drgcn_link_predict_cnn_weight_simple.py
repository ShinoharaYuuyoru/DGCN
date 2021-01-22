import argparse
import os
import load_data
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from drgcn_model_cnn import DRGCN
import drgcn_utils_cnn

import matplotlib.pyplot as plt

    
class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_rgcn_hidden_layers=1, #num_dkrl_hidden_layers=1,
                 dropout=0, use_cuda=False, reg_param=0, 
                 descDict={}, descWordNumDict={},
                 wordDict=dict(), wordNum=0,
                 sampledDescWordNumMax=100):
        super(LinkPredict, self).__init__()

        # drgcn model
        self.drgcn = DRGCN(num_nodes=in_dim, h_dim=h_dim, out_dim=h_dim, num_rels=num_rels * 2, num_bases=num_bases,
                         num_rgcn_hidden_layers=num_rgcn_hidden_layers, #num_dkrl_hidden_layers=num_dkrl_hidden_layers,
                         dropout=dropout, use_self_loop=False, use_cuda=use_cuda,
                         descDict=descDict, descWordNumDict=descWordNumDict,
                         wordDict=wordDict, wordNum=wordNum,
                         sampledDescWordNumMax=sampledDescWordNumMax)

        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))       # randomly init relation matrix
        nn.init.kaiming_uniform_(self.w_relation, nonlinearity='relu')      # xavier uniform performs not good with relu, so we use kaiming init.
        # nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))

        self.w_relation_inv = nn.Parameter(torch.Tensor(num_rels, h_dim))       # inverse relation weight
        nn.init.kaiming_uniform_(self.w_relation_inv, nonlinearity='relu')

        self.weight_matrix = nn.Parameter(torch.Tensor(in_dim, h_dim))
        nn.init.kaiming_uniform_(self.weight_matrix, nonlinearity='relu')

    def calc_score(self, h, r, t):
        def DistMult(h, r, t):
            score = torch.sum(h * r * t, dim=1)
            return score
        
        def TransE(h, r, t):
            # h = F.normalize(h, 2, -1)
            # r = F.normalize(r, 2, -1)
            # t = F.normalize(t, 2, -1)
            score = torch.norm((h + r - t), p=2, dim=1)        # L2
            return -score

        score = DistMult(h, r, t)       # DistMult
        # score = TransE(h, r, t)     # TransE

        return score

    def forward(self, g, h, r, norm, s_e_d_w_embeddings):
        return self.drgcn.forward(g, h, r, norm, s_e_d_w_embeddings)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))
    
    def getMixedEmbedding(self, rgcnEmbedding, dkrlEmbedding, node_id):
        weight_matrix = torch.sigmoid(self.weight_matrix)
        weight_matrix = weight_matrix[node_id.squeeze()]
        mix_embedding = rgcnEmbedding * weight_matrix * dkrlEmbedding

        return mix_embedding

    def get_loss(self, g, mixedEmbedding, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        
        # # PROBLEM: How to calculate the loss of joint model?
        # # mix_embedding = rgcnEmbedding*(1-mix_rate) + dkrlEmbedding*mix_rate
        # r = self.w_relation[triplets[:, 1]]
        # mix_h = mixedEmbedding[triplets[:, 0]]
        # mix_t = mixedEmbedding[triplets[:, 2]]
        # score = self.calc_score(mix_h, r, mix_t)

        # # score = 0
        # # # cnn - cnn
        # # score += self.calc_score(dkrlEmbedding[triplets[:, 0]], r, dkrlEmbedding[triplets[:, 2]])
        # # # cnn - rgcn
        # # score += self.calc_score(dkrlEmbedding[triplets[:, 0]], r, rgcnEmbedding[triplets[:, 2]])
        # # # rgcn - cnn
        # # score += self.calc_score(rgcnEmbedding[triplets[:, 0]], r, dkrlEmbedding[triplets[:, 2]])
        # # # rgcn - rgcn
        # # score += self.calc_score(rgcnEmbedding[triplets[:, 0]], r, rgcnEmbedding[triplets[:, 2]])
        
        # predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        # reg_loss = 0
        # # reg_loss += self.regularization_loss(dkrlEmbedding)
        # # reg_loss += self.regularization_loss(rgcnEmbedding)
        # reg_loss = self.regularization_loss(mixedEmbedding)

        # SimplE
        r = self.w_relation[triplets[:, 1]]
        r_inv = self.w_relation_inv[triplets[:, 1]]
        mix_h = mixedEmbedding[triplets[:, 0]]
        mix_t = mixedEmbedding[triplets[:, 2]]
        score = self.calc_score(mix_h, r, mix_t)
        score += self.calc_score(mix_h, r_inv, mix_t)
        score = score / 2
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = (torch.mean(mix_h.pow(2)) + torch.mean(mix_t.pow(2)) + torch.mean(r.pow(2)) + torch.mean(r_inv.pow(2))) / 4

        return predict_loss + self.reg_param * reg_loss

def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']

def createWordDict(descDict):
    wordDict = dict()
    wordId = 0      # 0 for 0 padding in embedding.

    for entity in descDict:
        desc = descDict[entity]
        desc = desc.split()
        for word in desc:
            if word not in wordDict:
                wordDict[word] = wordId
                wordId += 1
    
    return wordDict, len(wordDict)

def processDescDict(descDict, wordDict):
    newDescDict = {}
    for entity in descDict:
        desc = descDict[entity]
        desc = desc.split()
        descWordId = np.zeros(len(desc), dtype=np.int32)
        for idx, word in enumerate(desc):
            descWordId[idx] = wordDict[word] + 1        # 0 for padding
        newDescDict[entity] = descWordId
    
    return newDescDict

def getSampledDescWordList(node_id, descDict, sampledDescWordNumMax):
    sampledDescWordList = np.zeros((len(node_id), sampledDescWordNumMax), dtype=np.int32)

    for i, entity in enumerate(node_id):
        desc = descDict[entity]
        if len(desc) <= sampledDescWordNumMax:
            sampledDescWordList[i][:len(desc)] = desc
        else:
            sampledDescWordList[i][:] = desc[:sampledDescWordNumMax]
    
    return sampledDescWordList

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

    # Process descDict(full of text) to descDict(full of wordId)
    descDict = processDescDict(descDict, wordDict)

    # descWordNum statistic
    #   As we tested, in FB15K-237 the length of description is in [0, 704]
    #   704 is too big to process, so we should cut it to a proper length.
    #   The statistic shows, that 99% descriptions shorter than 331.
    #   So we decided to cut/pad all descriptions to the size 350.
    # maxDescWordNum = max(descWordNumDict.values())
    # sta = np.zeros(maxDescWordNum + 1, dtype=np.int32)
    # for val in descWordNumDict.values():
    #     sta[val] += 1       # MAX: FB15K-237 704, WN18RR 91
    # s = 0
    # for idx, a in enumerate(sta):
    #     if s > (sum(sta)*0.99):
    #         print(idx)      # FB15K-237: 331, WN18RR:40
    #         break
    #     else:
    #         s += a
    sampledDescWordNumMax = args.desc_word_num     # How many words in descriptions should be cut or padded to.

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create model
    model = LinkPredict(in_dim=num_nodes, h_dim=args.n_hidden, num_rels=num_rels, num_bases=args.n_bases,
                        num_rgcn_hidden_layers=args.n_rgcn_layers, #num_dkrl_hidden_layers=args.n_dkrl_layers,
                        dropout=args.dropout, use_cuda=use_cuda, reg_param=args.regularization,
                        descDict=descDict, descWordNumDict=descWordNumDict,
                        wordDict=wordDict, wordNum=wordNum,
                        sampledDescWordNumMax=sampledDescWordNumMax)
    
    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    # build test graph, which is the whole graph of train.txt
    # No need to convert to cuda(), because evaluation is running in CPU.
    test_graph, test_rel, test_norm = drgcn_utils_cnn.build_test_graph(num_nodes, num_rels, train_data)
    test_deg = test_graph.in_degrees(range(test_graph.number_of_nodes())).float().view(-1,1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))
    allDescWordList = getSampledDescWordList([entityNum for entityNum in range(num_nodes)], descDict, sampledDescWordNumMax)
    allDescWordList = torch.from_numpy(allDescWordList).long()      # Conver to long tensor

    # build adj list and calculate degrees for sampling
    adj_list, degrees = drgcn_utils_cnn.get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    learningRate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    model_state_file = 'drgcn_'+args.dataset+'_model_state.pth'
    forward_time = []
    backward_time = []

    best_mrr = 0

    # Load existing checkpoint.
    if os.path.exists(model_state_file):
        try:
            checkpoint = torch.load(model_state_file)
            model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError:
            print("Checkpoint unmatch! Retrain the model!")
        else:
            print("Existing checkpoint loaded.")
            if use_cuda:
                model.cpu()
            model.eval()
            print("Evaluating existing model...")
            with torch.no_grad():
                rgcnEmbedding, dkrlEmbedding = model(test_graph, test_node_id, test_rel, test_norm, allDescWordList)
            mixedEmbedding = model.getMixedEmbedding(rgcnEmbedding, dkrlEmbedding, test_node_id)
            best_mrr = drgcn_utils_cnn.calc_mrr(mixedEmbedding, model.w_relation, torch.LongTensor(train_data),
                                    valid_data, test_data, hits=[1, 3, 10], eval_p=args.eval_protocol)
                
    if use_cuda:
        model.cuda()

    # training loop
    print("start training...")
    epoch = 0
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
            drgcn_utils_cnn.generate_sampled_graph_and_labels(
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

        # Get sampled entitiy descriptions and split the words.
        sampledDescWordList = getSampledDescWordList(node_id.flatten().tolist(), descDict, sampledDescWordNumMax)
        sampledDescWordList = torch.from_numpy(sampledDescWordList).long()     # Convert to tensor

        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
            data, labels = data.cuda(), labels.cuda()
            g = g.to(args.gpu)
            sampledDescWordList = sampledDescWordList.cuda()

        # Calculate loss and backward
        #   Problem: How to get the loss within rgcnEmbedding and dkrlEmbedding?
        t0 = time.time()
        rgcnEmbedding, dkrlEmbedding = model(g, node_id, edge_type, edge_norm, sampledDescWordList)
        mixedEmbedding = model.getMixedEmbedding(rgcnEmbedding, dkrlEmbedding, node_id)
        loss = model.get_loss(g, mixedEmbedding, data, labels)
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
            with torch.no_grad():
                rgcnEmbedding, dkrlEmbedding = model(test_graph, test_node_id, test_rel, test_norm, allDescWordList)
            mixedEmbedding = model.getMixedEmbedding(rgcnEmbedding, dkrlEmbedding, test_node_id)
            mrr = drgcn_utils_cnn.calc_mrr(mixedEmbedding, model.w_relation, torch.LongTensor(train_data),
                                 valid_data, test_data, hits=[1, 3, 10], eval_p=args.eval_protocol)
            # save best model
            if mrr < best_mrr:
                if epoch >= args.n_epochs:
                    break
                learningRate = learningRate * 0.5
                for g in optimizer.param_groups:
                    g['lr'] = learningRate
                print("Learning rate changed to: " + str(learningRate))
            else:
                learningRate = learningRate * 0.9
                for g in optimizer.param_groups:
                    g['lr'] = learningRate
                print("Learning rate changed to: " + str(learningRate))
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
    with torch.no_grad():
        rgcnEmbedding, dkrlEmbedding = model(test_graph, test_node_id, test_rel, test_norm, allDescWordList)
    mixedEmbedding = model.getMixedEmbedding(rgcnEmbedding, dkrlEmbedding, test_node_id)
    mrr = drgcn_utils_cnn.calc_mrr(mixedEmbedding, model.w_relation, torch.LongTensor(train_data),
                            valid_data, test_data, hits=[1, 3, 10], eval_p=args.eval_protocol)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DGCN')

    # Required settings.
    #   When you use VSCode Python debugger to debug, check the launch.json configuration file.
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use. 'FB15K-237' or 'WN18RR'.")
    parser.add_argument("--gpu", type=int, required=True, help="GPU number(>=0) to use. Input '-1' to use CPU.")
    parser.add_argument("--eval_protocol", type=str, required=True, help="Type of evaluation probocol: 'raw' or 'filtered'.")
    parser.add_argument("--desc_word_num", type=int, required=True, help="Word number in each description to cut or pad. Recommend: FB15K-237 350, WN18RR 100.")

    # Optional settings.
    #   About the model
    #       Overall
    parser.add_argument("--n_hidden", type=int, default=100, help="Number of embedding size.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability.")
    parser.add_argument("--negative_sample", type=int, default=10, help="Number of negative samples per positive sample.")
    parser.add_argument("--grad_norm", type=float, default=1.0, help="Norm to clip gradient to.")
    parser.add_argument("--regularization", type=float, default=1.0, help="Regularization weight.")
    # parser.add_argument("--embedding_mix_rate", type=float, default=0.2, help="mix_embedding = rgcnEmbedding*(1-mix_rate) + dkrlEmbedding*mix_rate")
    #       RGCN
    parser.add_argument("--n_bases", type=int, default=20, help="Number of weight blocks for each relation.")
    parser.add_argument("--n_rgcn_layers", type=int, default=3, help="Number of RGCN layers / propagation rounds.")
    parser.add_argument("--graph_split_size", type=float, default=0.5, help="Portion of edges used as positive sample.")
    parser.add_argument("--edge_sampler", type=str, default="uniform", help="Type of edge sampler: 'uniform' or 'neighbor'.")
    #       DKRL
    # parser.add_argument("--n_dkrl_layers", type=int, default=1, help="Number of DKRL layers / propagation rounds.")

    #   About training.
    parser.add_argument("--n_epochs", type=int, default=10000, help="Number of minimum training epochs.")
    parser.add_argument("--graph_batch_size", type=int, default=1000, help="Number of edges to sample in each iteration.")
    parser.add_argument("--evaluate_every", type=int, default=500, help="Perform evaluation every n epochs.")
    #   About evaluating.
    # parser.add_argument("--eval_batch_size", type=int, default=500, help="Batch size when evaluating.")     # No use because we evalute one by one now.

    args = parser.parse_args()
    print(args)

    main(args)
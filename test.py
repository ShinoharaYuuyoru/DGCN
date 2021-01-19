# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# a=torch.Tensor([[[1,2,3,4],[5,6,7,8]],      # entity
#                 [[9,10,11,12],[13,14,15,16]],       # entity
#                 [[17,18,19,20],[21,22,23,24]]       # entity
#                 ])
# print(a)
# a=a.permute(0,2,1)
# print(a)
# # a=F.pad(input=a,pad=(2,1),mode='circular')
# # print(a)

# conv1=nn.Conv1d(in_channels=4, out_channels=4, kernel_size=2, padding_mode='circular')
# out=conv1(a)
# print(out)
# print(out.size())


import torch
import torch.nn as nn
import torch.nn.functional as F

# max_sent_len=35, batch_size=50, embedding_size=300
input = torch.randn(50, 35, 500)        # 50 entity per batch, 35 words in 1 entity, 1 word with 300-dim embedding
input = input.permute(0, 2, 1)

windows_size = [2,3,4]

all_features = []
for window_size in windows_size:
    features = F.pad(input, (0, window_size-1), mode='circular')

    conv = nn.Conv1d(in_channels=500, out_channels=500, kernel_size=window_size)
    act = nn.ReLU()
    maxpool = nn.MaxPool1d(kernel_size=35-window_size+1)

    features = conv(features)
    features = act(features)
    features = maxpool(features)

    all_features.append(features)

all_features = torch.cat(all_features, dim=1)
print(all_features)
print(all_features.size())

dropout = nn.Dropout(p=0.2)
fc = nn.Linear(in_features=500*len(windows_size), out_features=500)

all_features = all_features.flatten(start_dim=1)
all_features = dropout(all_features)
e_cnn_vec = fc(all_features)
print(e_cnn_vec)
print(e_cnn_vec.size())

# input = F.pad(input, (0, window_size-1), mode='circular')
# print(input)

# # batch_size x max_sent_len x embedding_size -> batch_size x embedding_size x max_sent_len
# print("input:", input.size())

# output = conv1(input)
# print(output)
# act = nn.ReLU()
# output = act(output)
# print(output)
# print("output:", output.size())
# # 最大池化
# pool1d = 
# pool1d_value = pool1d(output)
# print("最大池化输出：", pool1d_value.size())
# # 全连接
# fc = nn.Linear(in_features=500, out_features=500)
# fc_inp = pool1d_value.view(-1, pool1d_value.size(1))
# print("全连接输入：", fc_inp.size())
# fc_outp = fc(fc_inp)
# print("全连接输出：", fc_outp.size())
# # softmax
# m = nn.Softmax()
# out = m(fc_outp)
# print("输出结果值：", out)
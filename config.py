"""
配置文件
"""

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_batchsize = 256
test_batch_size = 1000



max_len = 20
use_word = False
batch_size = 128
embedding_dim = 256
dropout = 0.5
hidden_size = 128
method = "concat"
beam_width = 3 #保存数据的总数

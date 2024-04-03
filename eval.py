"""
进行模型的评估
"""

import torch
import config
from torch import optim
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq
from dataset import data_loader as train_dataloader
from word_sequence import word_sequence
import jieba
import numpy as np


def eval():
    encoder = Encoder()
    decoder = Decoder()
    model = Seq2Seq(encoder,decoder).to(config.device)
    model.load_state_dict(torch.load("/root/autodl-tmp/seq2seq/model/model.pkl"))

    while True:
        _input = input("请输入：")
        inputs = jieba.cut(_input)
        input_list = list(inputs)
        input_length = torch.LongTensor(len(input_list)).to(config.device)
        inputs = torch.LongTensor([word_sequence.transform(i, max_len=config.max_len) for i in inputs]).to(config.device)
        indices = np.array(model.evaluation(inputs,input_length)).flatten()
        output = word_sequence.inverse_transform(indices)
        print("answer:", output)


if __name__ == '__main__':
    eval()



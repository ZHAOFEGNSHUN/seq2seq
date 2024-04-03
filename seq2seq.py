import torch
import torch.nn as nn
from decoder import Decoder
import config
from encoder import Encoder

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(Seq2Seq,self).__init__()
        self.encoder = Encoder().to(config.device)
        self.decoder = Decoder().to(config.device)

    def forward(self, input,target,input_length,target_length):
        encoder_outputs,encoder_hidden = self.encoder(input,input_length)
        decoder_outputs,decoder_hidden = self.decoder(encoder_hidden,target,target_length,encoder_outputs)
        return decoder_outputs,decoder_hidden

    def evaluation(self,inputs,input_length):
        if inputs.size(1) == 0:  # 检查输入序列是否为空
            return []  # 如果为空，直接返回空列表或其他默认值
        encoder_outputs,encoder_hidden = self.encoder(inputs,input_length)
        decoded_sentence = self.decoder.evaluatoin_beamsearch_heapq(encoder_hidden,encoder_outputs)
        return decoded_sentence
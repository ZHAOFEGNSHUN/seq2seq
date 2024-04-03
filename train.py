import torch
import config
from torch import optim
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq
from dataset import data_loader as train_dataloader
from word_sequence import word_sequence

encoder = Encoder()
decoder = Decoder()
model = Seq2Seq(encoder,decoder)

#device在config文件中实现
model.to(config.device)

print(model)


# 加载模型时指定将参数加载到 CPU 上
# model = torch.load("/home/u202220081002025/seq2seq/model/model.pkl", map_location=torch.device('cpu'))
optimizer =  optim.Adam(model.parameters())
# optimizer.load_state_dict(torch.load("/home/u202220081002025/seq2seq/model/optimizer.pkl"))
criterion= nn.NLLLoss(ignore_index=word_sequence.PAD,reduction="mean")

def get_loss(decoder_outputs,target):
    target = target.view(-1) #[batch_size*max_len]
    decoder_outputs = decoder_outputs.view(config.batch_size*config.max_len,-1)
    return criterion(decoder_outputs,target)


def train(epoch):
    for idx,(input,target,input_length,target_len) in enumerate(train_dataloader):
        input = input.to(config.device)
        target = target.to(config.device)
        input_length = input_length.to(config.device)
        target_len = target_len.to(config.device)

        optimizer.zero_grad()
        ##[seq_len,batch_size,vocab_size] [batch_size,seq_len]
        decoder_outputs,decoder_hidden = model(input,target,input_length,target_len)
        loss = get_loss(decoder_outputs,target)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, idx * len(input), len(train_dataloader.dataset),
                   100. * idx / len(train_dataloader), loss.item()))

        torch.save(model.state_dict(), "/home/u202220081002025/seq2seq/model/model.pkl")
        torch.save(optimizer.state_dict(), '/home/u202220081002025/seq2seq/model/optimizer.pkl')

if __name__ == '__main__':
    for i in range(1):
        train(i)
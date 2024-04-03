import torch
import torch.nn as nn
import config
import random
import torch.nn.functional as F
from word_sequence import word_sequence
from attention import Attention
import heapq

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.max_seq_len = config.max_len
        self.vocab_size = len(word_sequence)
        self.embedding_dim = config.embedding_dim
        self.dropout = config.dropout

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.embedding_dim,padding_idx=word_sequence.PAD)
        self.gru = nn.GRU(input_size=self.embedding_dim,
                          hidden_size=config.hidden_size,
                          num_layers=1,
                          batch_first=True,
                          dropout=self.dropout)
        self.log_softmax = nn.LogSoftmax()

        self.fc = nn.Linear(config.hidden_size,self.vocab_size)
        self.attn = Attention(config.method,config.batch_size,config.hidden_size)
        self.Wa = nn.Linear(config.hidden_size*2,config.hidden_size,bias=False)

    def forward(self, encoder_hidden,target,target_length,encoder_outputs):
        # encoder_hidden [batch_size,hidden_size]
        # target [batch_size,seq-len]

        decoder_input = torch.LongTensor([[word_sequence.SOS]]*config.batch_size).to(config.device)
        decoder_outputs = torch.zeros(config.batch_size,config.max_len,self.vocab_size).to(config.device) #[batch_size,seq_len,14]

        decoder_hidden = encoder_hidden #[batch_size,hidden_size]

        for t in range(config.max_len):
            decoder_output_t , decoder_hidden, _ = self.forward_step(decoder_input,decoder_hidden,encoder_outputs)
            decoder_outputs[:,t,:] = decoder_output_t
            value, index = torch.topk(decoder_output_t, 1) # index [batch_size,1]
            decoder_input = index
        return decoder_outputs,decoder_hidden

    # def forward_step(self,decoder_input,decoder_hidden):
    #     """
    #     :param decoder_input:[batch_size,1]
    #     :param decoder_hidden: [1,batch_size,hidden_size]
    #     :return: out:[batch_size,vocab_size],decoder_hidden:[1,batch_size,didden_size]
    #     """
    #     embeded = self.embedding(decoder_input)  #embeded: [batch_size,1 , embedding_dim]
    #     out,decoder_hidden = self.gru(embeded,decoder_hidden) #out [1, batch_size, hidden_size]
    #     out = out.squeeze(0)
    #     out = F.log_softmax(self.fc(out),dim=-1)#[batch_Size, vocab_size]
    #     out = out.squeeze(1)
    #     # print("out size:",out.size(),decoder_hidden.size())
    #     return out,decoder_hidden
    def forward_step(self,decoder_input,decoder_hidden,encoder_outputs):
        """
        :param decoder_input:[batch_size,1]
        :param decoder_hidden: [1,batch_size,hidden_size]
        :param encoder_outputs: encoder中所有的输出，[batch_size,seq_len,hidden_size]
        :return: out:[batch_size,vocab_size],decoder_hidden:[1,batch_size,didden_size]
        """
        embeded = self.embedding(decoder_input)  #embeded: [batch_size,1 , embedding_dim]
        
        #TODO 可以把embeded的结果和前一次的context（初始值为全0tensor） concate之后作为结果
        #rnn_input = torch.cat((embeded, last_context.unsqueeze(0)), 2)
        
        # gru_out:[256,1, 128]  decoder_hidden: [1, batch_size, hidden_size]
        gru_out,decoder_hidden = self.gru(embeded,decoder_hidden)
        gru_out = gru_out.squeeze(1)
        
        #TODO 注意：如果是单层，这里使用decoder_hidden没问题（output和hidden相同）
        # 如果是多层，可以使用GRU的output作为attention的输入
        #开始使用attention
        attn_weights = self.attn(decoder_hidden,encoder_outputs)
        # attn_weights [batch_size,1,seq_len] * [batch_size,seq_len,hidden_size]
        context = attn_weights.bmm(encoder_outputs) #[batch_size,1,hidden_size]

        gru_out = gru_out.squeeze(0)  # [batch_size,hidden_size]
        context = context.squeeze(1)  # [batch_size,hidden_size]
        #把output和attention的结果合并到一起
        concat_input = torch.cat((gru_out, context), 1) #[batch_size,hidden_size*2]
        
        concat_output = torch.tanh(self.Wa(concat_input)) #[batch_size,hidden_size]

        output = F.log_softmax(self.fc(concat_output),dim=-1) #[batch_Size, vocab_size]
        # out = out.squeeze(1)
        return output,decoder_hidden,attn_weights
    
    # decoder中的新方法
    def evaluatoin_beamsearch_heapq(self,encoder_outputs,encoder_hidden):
        """使用 堆 来完成beam search，对是一种优先级的队列，按照优先级顺序存取数据"""

        batch_size = encoder_hidden.size(1)
        #1. 构造第一次需要的输入数据，保存在堆中
        decoder_input = torch.LongTensor([[word_sequence.SOS] * batch_size]).to(config.device)
        decoder_hidden = encoder_hidden #需要输入的hidden

        prev_beam = Beam()
        prev_beam.add(1,False,[decoder_input],decoder_input,decoder_hidden)
        while True:
            cur_beam = Beam()
            #2. 取出堆中的数据，进行forward_step的操作，获得当前时间步的output，hidden
            #这里使用下划线进行区分
            for _probility,_complete,_seq,_decoder_input,_decoder_hidden in prev_beam:
                #判断前一次的_complete是否为True，如果是，则不需要forward
                #有可能为True，但是概率并不是最大
                if _complete == True:
                    cur_beam.add(_probility,_complete,_seq,_decoder_input,_decoder_hidden)
                else:
                    decoder_output_t, decoder_hidden,_ = self.forward_step(_decoder_input, _decoder_hidden,encoder_outputs)
                    value, index = torch.topk(decoder_output_t, config.beam_width)  # [batch_size=1,beam_widht=3]
                #3. 从output中选择topk（k=beam width）个输出，作为下一次的input
                for m, n in zip(value[0], index[0]):
                    decoder_input = torch.LongTensor([[n]]).to(config.device)
                    seq = _seq + [n]
                    probility = _probility * m
                    if n.item() == word_sequence.EOS:
                        complete = True
                    else:
                        complete = False

                    #4. 把下一个实践步骤需要的输入等数据保存在一个新的堆中
                        cur_beam.add(probility,complete,seq,
                                    decoder_input,decoder_hidden)
            #5. 获取新的堆中的优先级最高（概率最大）的数据，判断数据是否是EOS结尾或者是否达到最大长度，如果是，停止迭代
            best_prob,best_complete,best_seq,_,_ = max(cur_beam)
            if best_complete == True or len(best_seq)-1 == config.max_len: #减去sos
                return self._prepar_seq(best_seq)
            else:
            #6. 则重新遍历新的堆中的数据
                prev_beam = cur_beam
                                        
    def _prepar_seq(self,seq):#对结果进行基础的处理，共后续转化为文字使用
        if seq[0].item() == word_sequence.SOS:
            seq=  seq[1:]
        if  seq[-1].item() == word_sequence.EOS:
            seq = seq[:-1]
        seq = [i.item() for i in seq]
        return seq



class Beam:
    def __init__(self):
        self.heap = list() #保存数据的位置
        self.beam_width = config.beam_width #保存数据的总数

    def add(self,probility,complete,seq,decoder_input,decoder_hidden):
        """
        添加数据，同时判断总的数据个数，多则删除
        :param probility: 概率乘积
        :param complete: 最后一个是否为EOS
        :param seq: list，所有token的列表
        :param decoder_input: 下一次进行解码的输入，通过前一次获得
        :param decoder_hidden: 下一次进行解码的hidden，通过前一次获得
        :return:
        """
        heapq.heappush(self.heap,[probility,complete,seq,decoder_input,decoder_hidden])
        #判断数据的个数，如果大，则弹出。保证数据总个数小于等于3
        if len(self.heap)>self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):#让该beam能够被迭代
        return iter(self.heap)


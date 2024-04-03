import torch.nn as nn
import torch
import torch.nn.functional as F
from word_sequence import word_sequence
import config

class Attention(nn.Module):
    def __init__(self,method,batch_size,hidden_size):
        super(Attention,self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        assert self.method in ["dot","general","concat"],"method 只能是 dot,general,concat,当前是{}".format(self.method)

        if self.method == "dot":
            pass
        elif self.method == "general":
            self.Wa = nn.Linear(hidden_size,hidden_size,bias=False)
        elif self.method == "concat":
            self.Wa = nn.Linear(hidden_size*2,hidden_size,bias=False)
            self.Va = nn.Parameter(torch.FloatTensor(batch_size,hidden_size))

    def forward(self, hidden,encoder_outputs):
        """
        :param hidden:[1,batch_size,hidden_size]
        :param encoder_outputs: [batch_size,seq_len,hidden_size]
        :return:
        """
        batch_size,seq_len,hidden_size = encoder_outputs.size()

        hidden = hidden.squeeze(0) #[batch_size,hidden_size]

        if self.method == "dot":
            return self.dot_score(hidden,encoder_outputs)
        elif self.method == "general":
            return self.general_score(hidden,encoder_outputs)
        elif self.method == "concat":
            return self.concat_score(hidden,encoder_outputs)

    def _score(self,batch_size,seq_len,hidden,encoder_outputs):
        # 速度太慢
        # [batch_size,seql_len]
        attn_energies = torch.zeros(batch_size,seq_len).to(config.device)
        for b in range(batch_size):
            for i in range(seq_len):
                #encoder_output : [batch_size,seq_len,hidden_size]
                #deocder_hidden :[batch_size,hidden_size]
                #torch.Size([256, 128]) torch.Size([128]) torch.Size([256, 24, 128]) torch.Size([128])
                # print("attn size:",hidden.size(),hidden[b,:].size(),encoder_output.size(),encoder_output[b,i].size())
                    attn_energies[b,i] = hidden[b,:].dot(encoder_outputs[b,i]) #dot score
        return F.softmax(attn_energies).unsqueeze(1)  # [batch_size,1,seq_len]

    def dot_score(self,hidden,encoder_outputs):
        """
        dot attention
        :param hidden:[batch_size,hidden_size] --->[batch_size,hidden_size,1]
        :param encoder_outputs: [batch_size,seq_len,hidden_size]
        :return:
        """
        #hiiden :[hidden_size] -->[hidden_size,1] ，encoder_output:[seq_len,hidden_size]
        
        
        hidden = hidden.unsqueeze(-1)
        attn_energies = torch.bmm(encoder_outputs, hidden)
        attn_energies = attn_energies.squeeze(-1) #[batch_size,seq_len,1] ==>[batch_size,seq_len]

        return F.softmax(attn_energies).unsqueeze(1)  # [batch_size,1,seq_len]

    def general_score(self,hidden,encoder_outputs):
        """
        general attenion
        :param batch_size:int
        :param hidden: [batch_size,hidden_size]
        :param encoder_outputs: [batch_size,seq_len,hidden_size]
        :return:
        """
        x = self.Wa(hidden) #[batch_size,hidden_size]
        x = x.unsqueeze(-1) #[batch_size,hidden_size,1]
        attn_energies = torch.bmm(encoder_outputs,x).squeeze(-1) #[batch_size,seq_len,1]
        return F.softmax(attn_energies,dim=-1).unsqueeze(1)      # [batch_size,1,seq_len]

    def concat_score(self,hidden,encoder_outputs):
        """
        concat attention
        :param batch_size:int
        :param hidden: [batch_size,hidden_size]
        :param encoder_outputs: [batch_size,seq_len,hidden_size]
        :return:
        """
        #需要先进行repeat操作，变成和encoder_outputs相同的形状,让每个batch有seq_len个hidden_size
        x = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1) # [batch_size, seq_len, hidden_size]
        # print("x:", x.size())
        # print("encoder_outputs:", encoder_outputs.size())
        x = torch.tanh(self.Wa(torch.cat([x,encoder_outputs],dim=-1))) #[batch_size,seq_len,hidden_size*2] --> [batch_size,seq_len,hidden_size]
        #va [batch_size,hidden_size] ---> [batch_size,hidden_size,1]
        attn_energis = torch.bmm(x,self.Va.unsqueeze(2))  #[batch_size,seq_len,1]
        attn_energis = attn_energis.squeeze(-1)
        # print("concat attention:",attn_energis.size(),encoder_outputs.size())
        return F.softmax(attn_energis,dim=-1).unsqueeze(1) #[batch_size,1,seq_len]
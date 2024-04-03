import torch
import config
from torch.utils.data import Dataset,DataLoader
from word_sequence import word_sequence


class ChatDataset(Dataset):
    def __init__(self):
        super(ChatDataset,self).__init__()

        input_path = "/home/u202220081002025/seq2seq/corpus/input.txt"
        target_path = "/home/u202220081002025/seq2seq/corpus/output.txt"
        if config.use_word:
            input_path = "/root/autodl-tmp/seq2seq/corpus/input_word.txt"
            target_path = "/root/autodl-tmp/seq2seq/corpus/output_word.txt"

        self.input_lines = open(input_path).readlines()
        self.target_lines = open(target_path).readlines()
        assert len(self.input_lines) == len(self.target_lines) ,"input和target文本的数量必须相同"
    def __getitem__(self, index):
        input = self.input_lines[index].strip().split()
        target = self.target_lines[index].strip().split()
        if len(input) == 0 or len(target)==0:
            input = self.input_lines[index+1].strip().split()
            target = self.target_lines[index+1].strip().split()
        #此处句子的长度如果大于max_len，那么应该返回max_len
        return input,target,min(len(input),config.max_len),min(len(target),config.max_len)

    def __len__(self):
        return len(self.input_lines)

def collate_fn(batch):
    #1.排序
    batch = sorted(batch,key=lambda x:x[2],reverse=True)
    input, target, input_length, target_length = zip(*batch)

    # 2.进行padding的操作
    input = torch.LongTensor([word_sequence.transform(i, max_len=config.max_len) for i in input])
    target = torch.LongTensor([word_sequence.transform(i, max_len=config.max_len, add_eos=True) for i in target])
    input_length = torch.LongTensor(input_length)
    target_length = torch.LongTensor(target_length)

    return input, target, input_length, target_length

data_loader = DataLoader(dataset=ChatDataset(),batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn,drop_last=True)

if __name__ == '__main__':
    for idx, (input, target, input_lenght, target_length) in enumerate(data_loader):
        print("idx:", idx)
        print("input:", input)
        print("taget:", target)
        print("input_lenght:", input_lenght)
        print("target_length:", target_length)
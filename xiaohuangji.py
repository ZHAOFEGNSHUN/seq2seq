from tqdm import tqdm
import jieba

def format_xiaohuangji_corpus(word=False):
    """处理小黄鸡的语料"""
    if word:
        corpus_path = "/root/autodl-tmp/seq2seq/corpus/xiaohuangji50w_nofenci.conv"
        input_path = "/root/autodl-tmp/seq2seq/corpus/input_word.txt"
        output_path = "/root/autodl-tmp/seq2seq/corpus/output_word.txt"
    else:
        corpus_path = "/root/autodl-tmp/seq2seq/corpus/xiaohuangji50w_nofenci.conv"
        input_path = "/root/autodl-tmp/seq2seq/corpus/input.txt"
        output_path = "/root/autodl-tmp/seq2seq/corpus/output.txt"


    f_input = open(input_path,"a")
    f_output = open(output_path,"a")
    pair = []
    for line in tqdm(open(corpus_path, encoding="utf-8"),ascii=True):
        if line.strip() == "E":
            if not pair:
                continue
            else:
                assert len(pair) == 2,"长度必须是2"
                if len(pair[0].strip())>=1 and len(pair[1].strip())>=1:
                    f_input.write(pair[0]+"\n")
                    f_output.write(pair[1]+"\n")
                pair = []
        elif line.startswith("M"):
            line = line[1:]
            if word:
                pair.append(" ".join(list(line.strip())))
            else:
                pair.append(" ".join(jieba.cut(line.strip())))
if __name__ == '__main__':
    format_xiaohuangji_corpus()
# word_sequence.py
import config
import pickle

class Word2Sequence():
    UNK_TAG = "UNK"
    PAD_TAG = "PAD"
    SOS_TAG = "SOS"
    EOS_TAG = "EOS"

    UNK = 0
    PAD = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {
            self.UNK_TAG :self.UNK,
            self.PAD_TAG :self.PAD,
            self.SOS_TAG :self.SOS,
            self.EOS_TAG :self.EOS
        }
        self.count = {}
        self.fited = False

    def to_index(self,word):
        """word -> index"""
        assert self.fited == True,"必须先进行fit操作"
        return self.dict.get(word,self.UNK)

    def to_word(self,index):
        """index -> word"""
        assert self.fited , "必须先进行fit操作"
        if index in self.inversed_dict:
            return self.inversed_dict[index]
        return self.UNK_TAG

    def __len__(self):
        return len(self.dict)

    def fit(self, sentence):
        """
        :param sentence:[word1,word2,word3]
        :param min_count: 最小出现的次数
        :param max_count: 最大出现的次数
        :param max_feature: 总词语的最大数量
        :return:
        """
        for a in sentence:
            if a not in self.count:
                self.count[a] = 0
            self.count[a] += 1

        self.fited = True

    def build_vocab(self, min_count=1, max_count=None, max_feature=None):

        # 比最小的数量大和比最大的数量小的需要
        if min_count is not None:
            self.count = {k: v for k, v in self.count.items() if v >= min_count}
        if max_count is not None:
            self.count = {k: v for k, v in self.count.items() if v <= max_count}

        # 限制最大的数量
        if isinstance(max_feature, int):
            count = sorted(list(self.count.items()), key=lambda x: x[1])
            if max_feature is not None and len(count) > max_feature:
                count = count[-int(max_feature):]
            for w, _ in count:
                self.dict[w] = len(self.dict)
        else:
            for w in sorted(self.count.keys()):
                self.dict[w] = len(self.dict)

        # 准备一个index->word的字典
        self.inversed_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence,max_len=None,add_eos=False):
        """
        实现吧句子转化为数组（向量）
        :param sentence:
        :param max_len:
        :return:
        """
        assert self.fited, "必须先进行fit操作"

        r = [self.to_index(i) for i in sentence]
        if max_len is not None:
            if max_len>len(sentence):
                if add_eos:
                    r+=[self.EOS]+[self.PAD for _ in range(max_len-len(sentence)-1)]
                else:
                    r += [self.PAD for _ in range(max_len - len(sentence))]
            else:
                if add_eos:
                    r = r[:max_len-1]
                    r += [self.EOS]
                else:
                    r = r[:max_len]
        else:
            if add_eos:
                r += [self.EOS]
        # print(len(r),r)
        return r

    def inverse_transform(self,indices):
        """
        实现从数组 转化为 向量
        :param indices: [1,2,3....]
        :return:[word1,word2.....]
        """
        sentence = []
        for i in indices:
            word = self.to_word(i)
            sentence.append(word)
        return sentence

#之后导入该word_sequence使用
word_sequence = pickle.load(open("/home/u202220081002025/seq2seq/model/ws.pkl","rb")) if not config.use_word else pickle.load(open("./pkl/ws_word.pkl","rb"))



if __name__ == '__main__':
    from word_sequence import Word2Sequence
    from tqdm import tqdm
    import pickle

    word_sequence = Word2Sequence()
    #词语级别
    input_path = "/root/autodl-tmp/seq2seq/corpus/input.txt"
    target_path = "/root/autodl-tmp/seq2seq/corpus/output.txt"
    for line in tqdm(open(input_path).readlines()):
        word_sequence.fit(line.strip().split())
    for line in tqdm(open(target_path).readlines()):
        word_sequence.fit(line.strip().split())
	
    #使用max_feature=5000个数据
    word_sequence.build_vocab(min_count=5,max_count=None,max_feature=5000)
    print(len(word_sequence))
    pickle.dump(word_sequence,open("/root/autodl-tmp/seq2seq/model/ws.pkl","wb"))
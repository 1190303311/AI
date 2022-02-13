from collections import defaultdict
from tqdm import tqdm

BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'
BOW_TOKEN = '<BOW>'
EOW_TOKEN = '<EOW>'


class Vocab:
    def __init__(self, text, min_freq=1, reserved_tokens=None):
        self.idx_to_token, self.token_to_idx = self.build(text, min_freq, reserved_tokens)

    def build(self, text, min_freq=1, reserved_tokens=None):
        # 创建词表，输入的text包含若干句子，每个句子由若干标记构成
        token_freqs = defaultdict(int)  # 存储标记及其出现次数的映射字典
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        # 无重复的标记，其中预留了未登录词（Unknown word）标记（<unk>）以及若干用户自定义的预留标记
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() if freq >=
                        min_freq and token != "<unk>"]
        idx2token = uniq_tokens
        token2idx = {v: k for k, v in enumerate(uniq_tokens)}
        return idx2token, token2idx

    def __len__(self):
        # 返回词表的大小，即词表中有多少个互不相同的标记
        return len(self.idx_to_token)

    def __getitem__(self, token):
        # 查找输入标记对应的索引值，如果该标记不存在，则返回标记<unk>的索引值（0）
        return self.token_to_idx.get(token, 0)

    def convert_tokens_to_ids(self, tokens):
        # 查找一系列输入标记对应的索引值
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        # 查找一系列索引值对应的标记
        return [self.idx_to_token[index] for index in indices]


def load_corpus(path, max_tok_len=None, max_seq_len=None):
    text = []
    charset = {BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, BOW_TOKEN, EOW_TOKEN}
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            tokens = line.rstrip().split(' ')
            if max_seq_len is not None and len(tokens) + 2 > max_seq_len:
                tokens = line[:max_seq_len - 2]
            sent = [BOS_TOKEN]
            for token in tokens:
                if max_tok_len is not None and len(token) + 2 > max_tok_len:
                    token = token[:max_tok_len - 2]
                sent.append(token)
                for ch in token:
                    charset.add(ch)
            sent.append(EOS_TOKEN)
            text.append(sent)
    vocab_w = Vocab(text=text, min_freq=2, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    vocab_c = Vocab(list(charset), min_freq=1)

    corpus_w = [vocab_w.convert_tokens_to_ids(sent) for sent in text]
    corpus_c = []
    bow = vocab_c[BOW_TOKEN]
    eow = vocab_c[EOW_TOKEN]
    for i, sent in enumerate(text):
        sent_c = []
        for token in sent:
            if token == BOS_TOKEN or token == EOS_TOKEN:
                token_c = [bow, vocab_c[token], eow]
            else:
                token_c = [bow] + vocab_c.convert_tokens_to_ids(token) + [eow]
            sent_c.append(token_c)
        corpus_c.append(sent_c)

    return corpus_w, corpus_c, vocab_w, vocab_c


def save_vocab(vocab, path):
    list = vocab.idx_to_token
    with open(path, 'w') as f:
        for token in list:
            f.write(token)
            f.write('\n')

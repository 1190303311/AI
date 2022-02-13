import json
import os

from torch.utils.data import Dataset
import torch
from utils import PAD_TOKEN
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F


class BiLMDataset(Dataset):
    def __init__(self, corpus_w, corpus_c, vocab_w, vocab_c):
        super(BiLMDataset, self).__init__()
        self.pad_w = vocab_w[PAD_TOKEN]
        self.pad_c = vocab_c[PAD_TOKEN]

        self.data = []
        for sent_w, sent_c in zip(corpus_w, corpus_c):
            self.data.append((sent_w, sent_c))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, examples):
        '''
        用于Dataloader指定函数collate_fn：
        train_loader = DataLoader(train_data, collate_fn=train_data.collate_fn, batch_size=configs.batch_size, shuffle=True)
        其中train_data是此类的实例，Dataloader会每次从self.data采样batch_size个，经过这个函数，将得到的输出封装成一个batch
        :param examples:self.data的batch个采样
        :return:
        inputs_w(batch_size, seq_len)
        inputs_c(batch_size, seq_len, max_tok_len)
        seq_lens(batch_size)
        targets_w、targets_c同上
        '''
        # ex = (sent_w, sent_c)
        seq_len = torch.LongTensor([len(ex[0]) for ex in examples])
        inputs_w = [torch.tensor(ex[0]) for ex in examples]
        # torch.nn.utils.rnn.pad_sequence, 输入要求是batch个tensor的list，按照最长的长度补齐
        # 返回(batch_size, max_seq_len)
        inputs_w = pad_sequence(inputs_w, batch_first=True, padding_value=self.pad_w)
        batch_size, max_seq_len = inputs_w.shape
        max_tok_len = max([max([len(tok) for tok in ex[1]]) for ex in examples])
        inputs_c = torch.LongTensor(batch_size, max_seq_len, max_tok_len).fill_(self.pad_c)
        for i, (sent_w, sent_c) in enumerate(examples):
            for j, tok in enumerate(sent_c):
                inputs_c[i][j][:len(tok)] = torch.LongTensor(tok)

        targets_fw = torch.LongTensor(inputs_w.shape).fill_(self.pad_w)
        targets_bw = torch.LongTensor(inputs_w.shape).fill_(self.pad_w)
        # 得到前向和后向语言模型的target
        for i, (sent_w, sent_c) in enumerate(examples):
            targets_fw[i][:len(sent_w) - 1] = torch.LongTensor(sent_w[1:])
            targets_fw[i][1:len(sent_w)] = torch.LongTensor(sent_w[:len(sent_w) - 1])

        return inputs_w, inputs_c, seq_len, targets_fw, targets_bw


class Highway(nn.Module):
    def __init__(self, input_dim, num_layers, activation=F.relu):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )

        self.activation = activation
        for layer in self.layers:  # ?
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        curr_inputs = inputs
        for layer in self.layers:
            projected_inputs = layer(curr_inputs)
            hidden = self.activation(projected_inputs[:, 0:self.input_dim])
            gate = torch.sigmoid(projected_inputs[:, self.input_dim:])
            curr_inputs = gate * curr_inputs + (1 - gate) * hidden
        return curr_inputs


class ConvTokenEmbedder(nn.Module):
    '''
    对每个单词（字符序列）进行卷积，得到独立的单词表示
    vocab_c:字符级词表
    char_embedding_dim:字符向量维度
    char_conv_filters:卷积核大小
    num_highways:Highway网络层数
    '''

    def __init__(self, vocab_c, char_embedding_dim, char_conv_filters, num_highways, output_dim, pad="<PAD>"):
        super(ConvTokenEmbedder, self).__init__()
        self.vocab_c = vocab_c
        self.char_embeddings = nn.Embedding(
            len(vocab_c),
            char_embedding_dim,
            padding_idx=vocab_c[pad]  # embedding词时，词中的pad会被embedding成0
        )
        self.char_embeddings.weight.data.uniform_(-0.25, 0.25)  # 均匀分布
        self.convolutions = nn.ModuleList()
        for kernel_size, out_channels in char_conv_filters:
            conv = torch.nn.Conv1d(
                in_channels=char_embedding_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=True
            )
            self.convolutions.append(conv)
        # 多个卷积网络结果拼接后的维度，对单独的词(字符序列)进行卷积，
        # embedding_dim作为输入通道，在单词字符序列长度维度上池化，每个卷积核（对应一个outchannel）输出一个维
        self.num_filters = sum(f[1] for f in char_conv_filters)
        self.num_highways = num_highways
        self.highways = Highway(self.num_filters, self.num_highways, activation=F.relu)
        #
        self.projection = nn.Linear(self.num_filters, output_dim, bias=True)

    def forward(self, inputs):
        batch_size, seq_len, token_len = inputs.shape
        inputs = inputs.view(batch_size * seq_len, -1)
        char_embeds = self.char_embeddings(inputs)  # (batch_size*seq_len, token_len, char_embedding_dim)
        char_embeds = char_embeds.transpose(1, 2)

        conv_hiddens = []
        for i in range(len(self.convolutions)):
            conv_hidden = self.convolutions[i](char_embeds)  # (batch_size*seq_len, out_channels[i], -1)
            conv_hidden, _ = torch.max(conv_hidden, dim=-1)  # pool, (batch_size*seq_len, out_channels[i])
            conv_hidden = F.relu(conv_hidden)
            conv_hiddens.append(conv_hidden)
        # 最后一维拼接
        token_embeds = torch.cat(conv_hiddens, dim=-1)  # (batch_size*seq_len, out_channels)
        token_embeds = self.highways(token_embeds)
        token_embeds = self.projection(token_embeds)
        token_embeds = token_embeds.view(batch_size, seq_len, -1)
        # (batch_size, seq_len, output_dim)
        return token_embeds


class ELMoLstmEncoder(nn.Module):
    '''
    双向LSTM解码器，获取序列每一时刻、每一层的前向、后向表示
    pytorch的lstm可以直接构建多层，但无法得到中间层的hidden表示
    用单层搭建
    '''

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ELMoLstmEncoder, self).__init__()
        # LSTM各中间层及输出层和输入表示层维度相同
        self.projection_dim = input_dim
        self.num_layers = num_layers

        # 前向LSTM
        self.forward_layers = nn.ModuleList()
        # 前向LSTM投射层：hidden_dim->self.projection_dim
        self.forward_projection = nn.ModuleList()
        # 后向LSTM
        self.backward_layers = nn.ModuleList()
        # 后向投射层
        self.backward_projection = nn.ModuleList()

        lstm_input_dim = input_dim
        for _ in range(num_layers):
            forward_layer = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=1, batch_first=True)
            forward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)
            backward_layer = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=1, batch_first=True)
            backward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)

            lstm_input_dim = self.projection_dim

            self.forward_layers.append(forward_layer)
            self.forward_projection.append(forward_projection)
            self.backward_layers.append(backward_layer)
            self.backward_projection.append(backward_projection)

    def forward(self, inputs, lengths):
        '''
        :param inputs:(batch_size, seq_len, input_dim)
        :param lengths: (batch_size)每个句子的真实长度
        :return:
        '''
        batch_size, seq_len, input_dim = inputs.shape
        # 构建反向lstm的输入
        # [1,2,3,0,0] -> [3,2,1,0,0]
        rev_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)  # repeat，对应维度重复对应次
        # rev_idx (batch_size, seq_len)
        for i in range(lengths.shape[0]):
            rev_idx[i, :lengths[i]] = torch.arange(lengths[i] - 1, -1, -1)  # (start, end, step)不含end
        rev_idx = rev_idx.unsqueeze(2).expand_as(inputs)
        rev_idx = rev_idx.to(inputs.device)
        rev_inputs = inputs.gather(1, rev_idx)

        forward_inputs, backward_inputs = inputs, rev_inputs
        # 保存每一层前后向隐层状态
        stacked_forward_states, stacked_backward_states = [], []
        for layer_index in range(self.num_layers):
            packed_forward_inputs = pack_padded_sequence(
                forward_inputs, lengths, batch_first=True, enforce_sorted=False
            )
            packed_backward_inputs = pack_padded_sequence(
                backward_inputs, lengths, batch_first=True, enforce_sorted=False
            )

            forward_layer = self.forward_layers[layer_index]
            packed_forward, _ = forward_layer(packed_forward_inputs)
            forward = pad_packed_sequence(packed_forward, batch_first=True)[0]
            forward = self.forward_projection[layer_index](forward)
            stacked_forward_states.append(forward)

            backward_layer = self.backward_layers[layer_index]
            packed_backward, _ = backward_layer(packed_backward_inputs)
            backward = pad_packed_sequence(packed_backward, batch_first=True)[0]
            backward = self.backward_projection[layer_index](backward)
            # 恢复至序列的原始顺序
            stacked_backward_states.append(backward.gather(1, rev_idx))

        return stacked_forward_states, stacked_backward_states


class BiLM(nn.Module):
    def __init__(self, configs, vocab_w, vocab_c):
        super(BiLM, self).__init__()
        self.dropout_prob = configs.dropout
        self.num_classes = len(vocab_w)
        # 词表示编码器
        self.token_embedder = ConvTokenEmbedder(
            vocab_c,
            configs.char_embedding_dim,
            configs.char_conv_filters,
            configs.num_highways,
            configs.projection_dim
        )
        # EMLo LSTM编码器
        self.encoder = ELMoLstmEncoder(
            configs.projection_dim,
            configs.hidden_dim,
            configs.num_layers
        )

        self.classifier = nn.Linear(configs.projection_dim, self.num_classes)

    def forward(self, inputs, lengths):
        '''
        :param inputs: (batch_size, seq_len, token_len)
        :param lengths: (batch_size)
        :return:
        '''
        token_embeds = self.token_embedder(inputs)  # (batch_size, seq_len, configs.projection_dim)
        token_embeds = F.dropout(token_embeds, self.dropout_prob)
        forward, backward = self.encoder(token_embeds, lengths)  # forward (num_layers, batch_size, seq_len, hidden_dim)
        # 取前后向LSTM最后一层的表示计算语言模型输出
        return self.classifier(forward[-1]), self.classifier(backward[-1])

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.token_embedder.state_dict(), os.path.join(path, 'token_embedder.pth'))
        torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pth'))


class ELMo(nn.Module):
    def __init__(self, model_dir):
        super(ELMo, self).__init__()
        self.configs = json.load(open(os.path.join(model_dir, 'configs.json')))
        self.vocab_c = read_vocab(os.path.join(model_dir, 'char.dir'))

        self.token_embedder = ConvTokenEmbedder(
            self.vocab_c,
            self.configs.char_embedding_dim,
            self.configs.char_conv_filters,
            self.configs.num_highways,
            self.configs.projection_dim
        )

        self.encoder = ELMoLstmEncoder(
            self.configs.projection_dim,
            self.configs.hidden_dim,
            self.configs.num_layers,
        )

        self.output_dim = self.configs.projection_dim

        self.load_pretrained(model_dir)

    def load_pretrained(self, path):
        self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pth')))
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pth')))

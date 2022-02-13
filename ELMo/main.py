# coding=utf-8
import json

import numpy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from utils import PAD_TOKEN, load_corpus, save_vocab
from config import Config
from model.ELMo import BiLMDataset, BiLM
import fire
import os
from tqdm import tqdm

def train(**kwargs):
    configs = Config()
    configs.update(**kwargs)
    print('当前设置为：\n', configs)
    print('loading corpus')
    corpus_w, corpus_c, vocab_w, vocab_c = load_corpus(configs.train_file)
    train_data = BiLMDataset(corpus_w, corpus_c, vocab_w, vocab_c)
    train_loader = DataLoader(train_data, collate_fn=train_data.collate_fn, batch_size=configs.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab_w[PAD_TOKEN], #忽略所有PAD_TOKEN处的预测损失
        reduction='sum'
    )
    model = BiLM(configs, vocab_w, vocab_c)
    if configs.use_cuda:
        model.cuda()
    optimizer = optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr = configs.learning_rate
    )
    #训练
    model.train()
    for epoch in range(configs.num_epoch):
        total_loss = 0
        total_tags = 0   #有效预测位置数，非PAD_TOKEN处的预测
        for batch in tqdm(train_loader, desc=f'Training Epoch {epoch}'):
            inputs_w, inputs_c, seq_lens, targets_fw, targets_bw = batch
            if configs.use_cuda:
                inputs_w, inputs_c, seq_lens, targets_fw, targets_bw = \
                inputs_w.cuda(), inputs_c.cuda(), seq_lens.cuda(), targets_fw.cuda(), targets_bw.cuda()

            optimizer.zero_grad()
            outputs_fw, outputs_bw = model(inputs_c, seq_lens)
            loss_fw = criterion(outputs_fw.view(-1, outputs_fw.shape[-1]), targets_fw.view(-1))
            loss_bw = criterion(outputs_bw.view(-1, outputs_bw.shape[-1]), targets_bw.view(-1))
            loss = (loss_fw + loss_bw) / 2.0
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), configs.clip_grad)
            optimizer.step()
            total_loss += loss_fw.item()
            total_tags += seq_lens.sum().item()

        #以前向语言模型的困惑度作为性能指标
        train_ppl = numpy.exp(total_loss / total_tags)
        print(f"Train PPL: {train_ppl:.2f}")

    model.save_pretrained(configs.save_path)
    json.dump(configs.__dict__, open(os.path.join(configs.save_path, 'configs.json'), 'w'))
    save_vocab(vocab_w, os.path.join(configs.save_path, 'word.dic'))
    save_vocab(vocab_c, os.path.join(configs.save_path, 'char.dic'))


if __name__ == '__main__':
    #fire.Fire()
    train()


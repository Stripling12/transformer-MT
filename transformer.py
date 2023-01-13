import os

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import re
from torch.autograd import Variable


def make_data(sentences, src_vocab, tgt_vocab):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        # print(sentences[i][0].split())
        enc_input = [src_vocab[n] for n in sentences[i][0].split()]  # enc_input 转换为数字形式 [1,2,3,4,0]
        dec_input = [tgt_vocab[n] for n in sentences[i][1].split()]  # dec_input 转换为数字形式
        dec_output = [tgt_vocab[n] for n in sentences[i][2].split()]  # dec_output 转换为数字形式

        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)
    # print(enc_inputs)
    # print(torch.LongTensor(enc_inputs).shape)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, item):
        return self.enc_inputs[item], self.dec_inputs[item], self.dec_outputs[item]


def get_position_encoding(n_positions, d_model):
    """
    :param n_positions:  number of position
    :param d_model: embedding dimension
    :return: position_encoding table
    """

    def calulate_theta(position, index):
        """
        计算公式
        :param position: 单词在句中的位置
        :param index:  词向量的维度序号，最大不超过dim/2  在这里是d_model/2
        :return: 计算sin或cos里的角度theta
        """
        return position / np.power(10000, 2 * (index // 2) / d_model)

    def per_position_embedding(position):
        """
        对其中一个位置进行编码，计算d_model维向量中的每个信息。
        :param position: [0,1,2,3,4,5,...]其中的一个
        :return: 假如第一次是0，就计算0这个位置的编码，编码为d_model维
        """
        return [calulate_theta(position, i) for i in range(d_model)]

    # 对所有的位置进行编码
    pos_embedding = np.array([per_position_embedding(pos) for pos in range(n_positions)])
    pos_embedding[:, 0::2] = np.sin(pos_embedding[:, 0::2])  # 双数位置进行sin计算
    pos_embedding[:, 1::2] = np.cos(pos_embedding[:, 1::2])  # 单数位置进行cos计算

    return torch.FloatTensor(pos_embedding)


def padding_mask(seq_q, seq_k):
    """
    padding P 进行mask操作 ， 相当于加一个一摸一样的矩阵， 加的这个矩阵在P位置是-inf 其他位置是0 ，以便于对他的softmax特别小
    :param seq_q: [batch_size,seq_len]  tensor类型
    :param seq_k: [batch_size,seq_len]
        seq_len 可以是src的len，也可以是tgt的len
        seq_q中的seq_len 和 seq_k中的seq_len可以不相等
    :return: 成功标记了Padding位置的矩阵 [batch_size, len_q, len_k]
    """
    batch_size, len_q = seq_q.size()  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]] torch.Size([2, 5])
    batch_size, len_k = seq_k.size()
    # print(seq_k.data)
    # seq_k = [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
    # seq_k = [[F, F, F, F, T], [F, F, F, F, T]]
    padding_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k]
    # print(padding_mask.shape)
    return padding_mask.expand(batch_size, len_q, len_k)


# print(padding_mask(enc_inputs, enc_inputs))
# print(padding_mask(enc_inputs, enc_inputs).shape)
def triangle_up_mask(seq):
    """
    :parameter seq: [batch_size,tgt_len]
    """
    return_shape = [seq.size(0), seq.size(1), seq.size(1)]
    triangle_up = np.triu(np.ones(return_shape), k=1)  # k=1指上三角位置上移一个位置
    mask_matrix = torch.from_numpy(triangle_up).byte()
    mask_matrix = mask_matrix.data.eq(1)
    # print(mask_matrix)
    # print(mask_matrix.shape)
    return mask_matrix  # [batch_size,tgt_len, tgt_len]


# print(padding_mask(dec_inputs, dec_inputs) + triangle_up_mask(dec_inputs))


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, d_q, d_k, d_v, n_heads, d_ff, n_layers=8):
        super(Encoder, self).__init__()
        self.word_embed_layer = nn.Embedding(src_vocab_size, d_model)  # src_vocab_size 词表长度
        self.pos_embed_layer = nn.Embedding.from_pretrained(get_position_encoding(src_vocab_size, d_model))
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_q, d_k, d_v, n_heads, d_ff) for i in range(n_layers)])

    def forward(self, enc_inputs, d_model, d_q, d_k, d_v):
        word_embed = self.word_embed_layer(enc_inputs)  # [batch_size, src_len, d_model]
        pos_embed = self.pos_embed_layer(enc_inputs)  # [1, 2, 3, 4, 0] 编码就是他的位置  [batch_size, src_len, d_model]
        enc_outputs = word_embed + pos_embed  # 最终的输入是位置编码和词嵌入相加的结果，这是第一次EncoderLayer输入的信息，因为后面需要循环，所以为了简单没有改名
        enc_self_attention_mask = padding_mask(enc_inputs, enc_inputs)  # 对padding的位置进行标记 以便于后期对相应的位置进行-inf赋值
        correlations = []  # 保存self_attention权重值(相关度)，便于后期画图
        for layer in self.layers:
            enc_outputs, enc_self_attention = layer(enc_outputs, enc_self_attention_mask, d_model, d_q, d_k, d_v)
            correlations.append(enc_self_attention)
        # correlations [n_layers, batch_size, n_heads, len_q, d_v], enc_outputs [batch_size, len_q, d_model]
        # print("enc_outputs", enc_outputs.shape)
        return enc_outputs, correlations


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attention = MultiHeadAttention(d_model, d_q, d_k, d_v, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, encoder_inputs, self_attention_mask, d_model, d_q, d_k, d_v):
        """
        :param encoder_inputs: [batch_size, seq_len, d_model]
        :param self_attention_mask: [batch_size, seq_len, seq_len]
        :return: enc_outputs [batch_size, len_q, d_model]
        """
        # enc_outputs [batch_size, len_q(src_len), d_model], correlation [batch_size, n_heads, len_q(src_len), len_k(src_len)]
        enc_outputs, correlation = self.enc_self_attention(encoder_inputs, encoder_inputs, encoder_inputs,
                                                           self_attention_mask, d_model, d_q, d_k, d_v)
        enc_outputs = self.feed_forward(enc_outputs, d_model)
        return enc_outputs, correlation


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_q * n_heads)  # 转化维度 [batch_size, len_q/seq_len, d_q*n_heads]
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(d_v * n_heads, d_model)  # 转换回相同的维度[batch_size, src_len, d_model], 残差连接才能相加 (W_O)

    def forward(self, q, k, v, mask, d_model, d_q, d_k, d_v, n_heads=8):
        """
=
        :param q: [batch_size, len_q, d_model]
        :param k: [batch_size, len_k, d_model]
        :param v: [batch_size, len_v, d_model]
        :param mask: padding mask 的矩阵 [batch_size, seq_len, seq_len]
        :return 残差连接后的结果进行LayerNomalization之后的结果， 返回相关度以便于后期画图  [batch_size, len_q, d_model]
        """
        res, batch_size_q = q, q.size(0)  # 用来残差连接
        batch_size_k = k.size(0)
        batch_size_v = v.size(0)
        # print("res-->", res.shape)
        # print("----------before------k_shape----------------", k.shape)
        # [batch_size, len_q(seq_len), d_model] -> [batch_size, len_q(seq_len), d_q*n_heads] -> [batch_size, len_q(seq_len), n_heads, d_q] -> [batch_size, n_heads, len_q(seq_len), d_q]
        q = self.W_Q(q).view(batch_size_q, -1, n_heads, d_q).transpose(1,
                                                                       2)  # q: [batch_size, n_heads, len_q(seq_len), d_q]
        k = self.W_K(k).view(batch_size_k, -1, n_heads, d_k).transpose(1,
                                                                       2)  # k: [batch_size, n_heads, len_k(seq_len), d_k]
        v = self.W_V(v).view(batch_size_v, -1, n_heads, d_v).transpose(1,
                                                                       2)  # v: [batch_size, n_heads, len_v(seq_len), d_v]

        # print("----------------k_shape----------------", k.shape)
        attention_mask = mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attention_mask [batch_size, n_heads, seq_len, seq_len]
        # print(attention_mask.shape)
        # print("decoder_outputs", q.shape, "encoder_outputs", k.shape)
        attention_value, correlation = SelfAttention()(q, k, v,
                                                       attention_mask,
                                                       d_k)  # attention_value [batch_size, n_heads, len_q(seq_len), d_v]
        attention_value = attention_value.transpose(1, 2).reshape(batch_size_k, -1, d_v * n_heads)  # 为了进入full connect层
        output = self.fc(attention_value)  # [batch_size, len_q(seq_len), d_model]
        # print('output-->', output.shape)
        # correlation [batch_size, n_heads, len_q, len_k]
        return nn.LayerNorm(d_model)(res + output), correlation


class SelfAttention(nn.Module):
    def __init__(self, attention_drop=0.0):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(attention_drop)

    def forward(self, q, k, v, mask, d_k, scale=None):
        """
        self_attention，相似度计算
        :param q: [batch_size, n_heads, len_q, d_q]
        :param k: [batch_size, n_heads, len_k, d_k]
        :param v: [batch_size, n_heads, len_v, d_v]
        :param mask: [batch_size, n_heads, seq_len, seq_len]  mask矩阵，Padding的位置上是true
        :return: attention_value, correlation(方便后期画图)
        """
        # print("q", q.shape, "k", k.shape)
        correlation = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k)  # [batch_size, n_heads, len_q, len_k]
        # print("correlation->", correlation.shape)
        # print("mask-->", mask.shape)
        correlation.masked_fill_(mask, -float(
            "inf"))  # 对padding位置置为-inf， [batch_size, n_heads, len_q, len_k], 要求correlation维度和mask维度一直一致
        # scores.masked_fill_(mask, 1e-9)
        # print("mask->", correlation)
        # scores.masked_fill_(mask, -np.inf)
        if scale:
            correlation = correlation * scale
        correlation = nn.Softmax(dim=-1)(correlation)  # [batch_size, n_heads, len_q, len_k(len_v)]
        # print("softmax->", correlation)
        correlation = self.dropout(correlation)
        attention_value = torch.matmul(correlation, v)  # [batch_size, n_heads, len_q, d_v]
        return attention_value, correlation


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, inputs, d_model):
        """
        :param inputs: [batch_size, seq_len, d_model]
        :return: 残差连接和LayerNormalization  [batch_size, seq_len, d_model]
        """
        res = inputs
        fc_output = self.fc(inputs)
        # print("encoder_outputs", nn.LayerNorm(d_model)(res + fc_output).shape)
        return nn.LayerNorm(d_model)(res + fc_output)


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, d_q, d_k, d_v, n_heads, d_ff, n_layers=8):
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(tgt_vocab_size, d_model)  # tgt_vocab_size 词表的长度
        self.pos_embedding = nn.Embedding.from_pretrained(get_position_encoding(tgt_vocab_size, d_model))
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_q, d_k, d_v, n_heads, d_ff) for i in range(n_layers)])

    def forward(self, encoder_inputs, encoder_outputs, decoder_inputs, d_model, d_q, d_k, d_v):
        """
        :param encoder_inputs: [batch_size, src_len]
        :param encoder_outputs: [batch_size, len_q, d_model]
        :param decoder_inputs: [batch_size, tgt_len]
        :return: decoder_outputs, dec_self_correlations, dec_enc_correlations
        """
        decoder_inputs = torch.tensor(decoder_inputs).to(torch.int64)  # 转变成LongTensor以输入Embedding层
        word_embed = self.word_embedding(decoder_inputs)  # [batch_size, tgt_len, d_model]
        pos_embed = self.pos_embedding(decoder_inputs)  # [batch_size, tgt_len, d_model]
        decoder_outputs = word_embed + pos_embed
        dec_dec_padding_mask = padding_mask(decoder_inputs, decoder_inputs)  # [batch_size, tgt_len, tgt_len]
        triangle_up_matrix = triangle_up_mask(decoder_inputs)  # 为了解决不应用RNN的问题，即当前时刻不可以看到之后的信息
        # print(triangle_up_matrix)
        dec_self_attention_mask = dec_dec_padding_mask + triangle_up_matrix
        # dec_self_attention_mask = torch.gt(dec_dec_padding_mask + triangle_up_matrix, 0) # [batch_size, tgt_len, src_len]

        dec_enc_attention_mask = padding_mask(decoder_inputs, encoder_inputs)  # [batch_size, tgt_len, src_len]
        # print("dec_self_attention_mask->", dec_self_attention_mask)
        dec_self_correlations, dec_enc_correlations = [], []
        for layer in self.layers:
            # print("-----------------------------------------------------------------------------------------")
            # print("dec_enc_attention_mask-->", dec_enc_attention_mask.shape)
            decoder_outputs, dec_self_attention_correlations, dec_enc_attention_correlations = layer(decoder_outputs,
                                                                                                     encoder_outputs,
                                                                                                     dec_self_attention_mask,
                                                                                                     dec_enc_attention_mask,
                                                                                                     d_model, d_q, d_k,
                                                                                                     d_v, )
            dec_enc_attention_mask = padding_mask(decoder_inputs, encoder_inputs)

            dec_self_correlations.append(dec_self_attention_correlations)
            dec_enc_correlations.append(dec_enc_attention_correlations)
        return decoder_outputs, dec_self_correlations, dec_enc_correlations


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, n_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.decoder_self_attention_layer = MultiHeadAttention(d_model, d_q, d_k, d_v)
        self.decoder_encoder_attention_layer = MultiHeadAttention(d_model, d_q, d_k, d_v)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, decoder_inputs, encoder_outputs, dec_self_attention_mask, dec_enc_attention_mask, d_model, d_q,
                d_k, d_v):
        """

        :param decoder_inputs: [batch_size, tgt_len, d_model]
        :param encoder_outputs: [batch_size, src_len, d_model]
        :param dec_self_attention_mask: [batch_size, tgt_len, tgt_len]
        :param dec_enc_attention_mask: [batch_size, tgt_len, src_len] [2,6,5]
        :return: decoder_outputs, dec_self_attention_correlations, dec_enc_attention_correlations
        """
        # correlations [batch_size, n_heads, len_q(tgt_len), len_k(tgt_len)], decoder_outputs [batch_size, len_q(tgt_len), d_model]
        decoder_outputs, dec_self_attention_correlations = self.decoder_self_attention_layer(decoder_inputs,
                                                                                             decoder_inputs,
                                                                                             decoder_inputs,
                                                                                             dec_self_attention_mask,
                                                                                             d_model, d_q, d_k, d_v, )
        # encoder_outputs作为k, v ,decoder_outputs作为q
        # correlations [batch_size, n_heads, len_q(tgt_len), len_k(src_len)], decoder_outputs [batch_size, len_q(tgt_len), d_model]
        # print(encoder_outputs.shape)
        # print(decoder_outputs.shape)
        decoder_outputs, dec_enc_attention_correlations = self.decoder_encoder_attention_layer(decoder_outputs,
                                                                                               encoder_outputs,
                                                                                               encoder_outputs,
                                                                                               dec_enc_attention_mask,
                                                                                               d_model, d_q, d_k, d_v)
        decoder_outputs = self.feed_forward(decoder_outputs, d_model)  # decoder_outputs [batch_size, tgt_len, d_model]
        return decoder_outputs, dec_self_attention_correlations, dec_enc_attention_correlations


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, d_q, d_k, d_v, n_heads, d_ff, n_layers=8):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, d_q, d_k, d_v, n_heads, d_ff)
        self.decoder = Decoder(tgt_vocab_size, d_model, d_q, d_k, d_v, n_heads, d_ff)
        self.fc = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, encoder_inputs, decoder_inputs, d_model, d_q, d_k, d_v):
        """
        :param encoder_inputs: [batch_size, src_len]
        :param decoder_inputs: [batch_size, tgt_len]
        :return: outputs [batch_size*tgt_len, tgt_vocab_size]
        """
        # correlations [n_layers, batch_size, n_heads, len_q, d_v], enc_outputs [batch_size, src_len, d_model]
        enc_outputs, enc_self_correlations = self.encoder(encoder_inputs, d_model, d_q, d_k, d_v)
        # decoder_outputs [batch_size, tgt_len, d_model]
        decoder_outputs, dec_self_correlations, dec_enc_correlations = self.decoder(encoder_inputs, enc_outputs,
                                                                                    decoder_inputs, d_model, d_q, d_k,
                                                                                    d_v)
        outputs = self.fc(decoder_outputs)  # [batch_size, tgt_len, tgt_vocab_size]
        return outputs.view(-1, outputs.size(-1)), enc_self_correlations, dec_self_correlations, dec_enc_correlations


# def greedy_decoder(model, encoder_inputs, max_len, start_symbol):
#     """
#     为了构造decoder_input
#     :param max_len: max_len of target_sentence
#     :param model: Transformer Model
#     :param encoder_inputs: [batch_size, src_len, d_model]
#     :param start_symbol:
#     :return: Decoder inputs
#     """
#     encoder_outputs, enc_self_correlations = model.encoder(encoder_inputs)
#     #print(type(start_symbol))
#     decoder_inputs = torch.zeros(1, 1).fill_(start_symbol).type_as(encoder_inputs.data)  # 元素转变为int型
#     #print(decoder_inputs.shape)
#     for i in range(max_len - 1):
#         decoder_outputs, _, _ = model.decoder(encoder_inputs, encoder_outputs, decoder_inputs)
#         prob = model.fc(decoder_outputs[:, -1])  # 获取到各个单词的概率
#         print(prob)
#         print(prob.shape)
#         prob = prob.squeeze(0).max(1, keepdim=False)[1]
#         #_, next_word = torch.max(prob.squeeze(0), dim=-1)  # 取概率值最大的word
#         next_word = prob.data[0]
#         #print(next_word)
#         #print(type(next_word))
#
#         #next_word = torch.zeros(1, 1).fill_(next_word.item().cpu().detach().numpy()).type_as(encoder_inputs.data)
#         #print(next_word.data[i].shape)
#         decoder_inputs = torch.cat([decoder_inputs, torch.zeros(1, 1).fill_(next_word.cpu().detach().numpy()[0])], dim=-1)  # 将之前的结果与最新结果拼接
#         #print("-----decoder_inputs------>", decoder_inputs.shape)
#     return decoder_inputs


def greedy_decoder(model, src, src_mask, max_len, start_symbol, d_model, d_q, d_k, d_v):
    # memory = model.encode(src, src_mask)
    encoder_outputs, enc_self_correlations = model.encoder(src, d_model, d_q, d_k, d_v)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        print(i)
        out, _, _ = model.decoder(src, encoder_outputs, ys, d_model, d_q, d_k, d_v)
        # out, _, _ = model.decoder(encoder_outputs, src_mask,
        #                    Variable(ys),
        #                    Variable(subsequent_mask(ys.size(1))
        #                             .type_as(src.data)))
        prob = model.fc(out[:, -1])  #
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    print('success')
    return ys


def prepare_data(path, lines):
    """
    :param path: 文件路径
    :param lines: 取前多少行
    :return: english sentences list, cs sentences list, max length of english list and cs list
    """
    en = []
    cs_dec_input = []
    cs_dec_output = []
    with open(path, 'r', encoding='UTF-8') as f:
        count = 0
        for line in f:
            line = line.strip().split('\t')
            en.append(line[0].lower())
            cs_dec_input.append("BOS " + line[1].lower())
            cs_dec_output.append(line[1].lower() + " EOS")
            count += 1
            if count == lines:
                break
    max_len_en = np.max([len(i.split()) for i in en])  # 109
    max_len_cs = np.max([len(i.split()) for i in cs_dec_output])  # 116
    # print(max_len_en, max_len_cs)
    return en, cs_dec_input, cs_dec_output, max_len_en, max_len_cs


def make_dict(src, tgt):
    """
    'P', 'BOS', 'EOS' 为补充标志
    :param src: 原句列表
    :param tgt: 目标句子列表 ， 两表长度一致
    :return: src_id2word, src_word2id, tgt_id2word, tgt_word2id ,
    """
    src_word_list = ['P', 'BOS', 'EOS']
    tgt_word_list = ['P', 'BOS', 'EOS']
    for i in range(len(src)):
        src_per_sentence_words = src[i].lower().split()  # 对每一话进行分割，取出每一个词
        tgt_per_sentence_words = tgt[i].lower().split()  # 对每一话进行分割，取出每一个词
        for word in src_per_sentence_words:
            if word not in src_word_list:
                src_word_list.append(word)
        for word in tgt_per_sentence_words:
            if word not in tgt_word_list:
                tgt_word_list.append(word)
    src_id2word = {i: w for i, w in enumerate(src_word_list)}  # source sentences word vocabulary
    src_word2id = {w: i for i, w in enumerate(src_word_list)}
    tgt_id2word = {i: w for i, w in enumerate(tgt_word_list)}  # target sentences word vocabulary
    tgt_word2id = {w: i for i, w in enumerate(tgt_word_list)}
    return src_id2word, src_word2id, tgt_id2word, tgt_word2id


def show_figure(loss, title, xlable, ylable):
    plt.figure(figsize=(15, 10))
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.plot(loss)
    plt.show()


def train(model, data, d_model, d_q, d_k, d_v, path=None):
    # print(sentences)
    # Transformer Parameters

    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    losses = []
    for epoch in range(15):
        batch_losses = []
        for enc_inputs, dec_inputs, dec_outputs in data:
            model_outputs, encoder_self_correlations, decoder_self_correlations, dec_encoder_correlations = model(
                enc_inputs,
                dec_inputs, d_model, d_q, d_k, d_v)
            loss = criterion(model_outputs, dec_outputs.view(-1))
            batch_losses.append(loss)
            print("Epoch", "%04d" % (epoch + 1), "loss=", "{:.7f}".format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(torch.mean(torch.stack(batch_losses)))
    if path:
        torch.save(model.state_dict(), path)  # ./save/model.pt
    return losses


def evaluate(model, data, src_word2id, tgt_id2word, tgt_word2id, path, d_model, d_q, d_k, d_v):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(path):
        # 加载模型
        model.load_state_dict(torch.load(path))
        # 开始预测
        print(">>>>>>> start evaluate")
        model.eval()
        input_sentence = data.lower().split()  # '20 m/min total power requirement'
        enc_inputs = torch.LongTensor([[src_word2id[i] for i in input_sentence]])
        out = greedy_decoder(model, enc_inputs, padding_mask(enc_inputs, enc_inputs), 120, tgt_word2id["BOS"], d_model,
                             d_q, d_k, d_v)
        pred, _, _, _ = model(enc_inputs[0].view(1, -1), out, d_model, d_q, d_k, d_v)
        pred = pred.data.max(1, keepdim=True)[1]
        tgt_pre_len = 0
        results = [tgt_id2word[n.item()] for n in pred.squeeze()]  # 获取预测的结果
        #print(results)
        for i in range(len(results)):  # 因为EOS是一句话的结尾，所以取E之前的句子
            if results[i] == 'EOS':
                tgt_pre_len = i
                break
        print(data, '-->', [results[i] for i in range(tgt_pre_len)])
        print("<<<<<<< finished evaluate")
    else:
        print("Error: pleas train before evaluate")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    en, cs_dec_input, cs_dec_output, max_len_en, max_len_cs = prepare_data('D:\\Desktop\\en-cs.txt', 2275)

    for i in range(len(en)):
        while en[i].split().__len__() < max_len_en:
            en[i] += ' P'
    for i in range(len(cs_dec_input)):
        while cs_dec_input[i].split().__len__() < max_len_cs:
            cs_dec_input[i] += ' P'
    for i in range(len(cs_dec_output)):
        while cs_dec_output[i].split().__len__() < max_len_cs:
            cs_dec_output[i] += ' P'
    # print(cs)

    src_id2word, src_word2id, tgt_id2word, tgt_word2id = make_dict(en, cs_dec_output)
    # print(src_word2id)
    sentences = []
    for i in range(len(en)):
        sentence = [en[i], cs_dec_input[i], cs_dec_output[i]]
        sentences.append(sentence)

    d_model = 512  # Embedding Size
    d_ff = 2048  # Feed Forward Dimension 先升到2048维在降回原维数
    d_k = d_q = d_v = 64  # dimension of K(=Q) , V
    n_layers = 6  # number of Encoder /of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention

    enc_inputs, dec_inputs, dec_outputs = make_data(sentences, src_word2id, tgt_word2id)[:2048]
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=16, shuffle=True)
    model = Transformer(len(src_word2id), len(tgt_word2id), d_model, d_q, d_k, d_v, n_heads, d_ff)
    train_losses = train(model, loader, d_model, d_q, d_k, d_v, path='./save/model.pt')
    show_figure(train_losses, title='train_loss', xlable='epoch', ylable='loss')
    sentence = '5223 Service activities incidental to air transportation'
    evaluate(model, sentence, src_word2id, tgt_id2word, tgt_word2id, './save/model.pt', d_model, d_q, d_k, d_v)


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # sentences = [
    #     # enc_input                dec_input            dec_output
    #     ['ich möchte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    #     ['ich möchte ein cola P', 'S i want a coke .', 'i want a coke . E'],
    #     ['ich möchte dich P P', 'S i love you . P', 'i love you . P E'],
    # ]
    # english_sentences = []
    # chinese_sentences = []
    # with open("D:\\Desktop\\en-zh.tmx", 'r', encoding='utf-8') as file:
    #     #file.readline()
    #     i = 0
    #     for line in file:
    #         #print(str(line))
    #         en = re.findall("<tuv xml:lang=\"en\"><seg>(.*?)</seg></tuv>", line)
    #         i += 1
    #         if len(en) > 0:
    #             english_sentences.extend(en)
    #         ch = re.findall("<tuv xml:lang=\"zh\"><seg>(.*?)</seg></tuv>", line)
    #         if len(ch) > 0:
    #             chinese_sentences.extend(ch)
    #         if i > 21:
    #             break
    # print(english_sentences.__len__())
    # print(chinese_sentences.__len__())

    # max_len_english = np.max([len(i.split()) for i in english_sentences])  # 422
    # max_len_chinese = np.max([len(i) for i in chinese_sentences])  # 8757
    # print(max_len_english, max_len_chinese)
    # for i in range(len(english_sentences)):
    #     while english_sentences[i].split().__len__() < 422:
    #         english_sentences[i] += ' P'
    #
    # chinese_dec_input = []
    # for i in range(len(chinese_sentences)):
    main()

    # print([i for i in chinese_sentences if len(i)==8757])
    # print([i for i in english_sentences if len(i.split())==422])

    """
    # Padding Should be Zero
    src_words_list = ['P']
    for sentence_index in range(len(sentences)):
        src_words_per_sentence = sentences[sentence_index][0].split()
        for src_word in src_words_per_sentence:
            if src_word not in src_words_list:
                src_words_list.append(src_word)
    # print(src_words_list)
    src_word2idx = {w: i for i, w in enumerate(src_words_list)}
    # print(src_word2idx)

    tgt_words_list = []
    for sentence_index in range(len(sentences)):
        tgt_words_per_sentence = (sentences[sentence_index][1] + sentences[sentence_index][2]).split()
        # print(tgt_words_per_sentence)
        for tgt_word in tgt_words_per_sentence:
            if tgt_word not in tgt_words_list:
                tgt_words_list.append(tgt_word)
    tgt_word2idx = {w: i for i, w in enumerate(tgt_words_list)}
    # print(tgt_word2idx)
    """

    # src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5, 'liebe': 6, 'dich': 7}
    # src_vocab_size = len(src_vocab)
    #
    # tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8, 'love': 9, 'you': 10}
    # tgt_vocab_size = len(tgt_vocab)
    #
    # # index->word 为了输出句子结果
    # idx2word = {i: w for i, w in enumerate(tgt_vocab)}

    # src_len = 5
    # tgt_len = 6

    # Transformer Parameters
    # d_model = 512  # Embedding Size
    # d_ff = 2048  # Feed Forward Dimension 先升到2048维在降回原维数
    # d_k = d_q = d_v = 64  # dimension of K(=Q) , V
    # n_layers = 6  # number of Encoder /of Decoder Layer
    # n_heads = 8  # number of heads in Multi-Head Attention

    # enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    # loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=2, shuffle=True)
    #
    # model = Transformer()
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    # for epoch in range(30):
    #     for enc_inputs, dec_inputs, dec_outputs in loader:
    #         model_outputs, encoder_self_correlations, decoder_self_correlations, dec_encoder_correlations = model(
    #             enc_inputs,
    #             dec_inputs)
    #         loss = criterion(model_outputs, dec_outputs.view(-1))
    #         print("Epoch", "%04d" % (epoch + 1), "loss=", "{:.7f}".format(loss))
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #
    # model.eval()
    # enc_inputs, _, _ = next(iter(loader))
    # dec_inputs = greedy_decoder(model, enc_inputs, padding_mask(enc_inputs, enc_inputs), max_len=tgt_vocab_size, start_symbol=tgt_vocab["S"])
    # print(enc_inputs)
    # pred, _, _, _ = model(enc_inputs[1].view(1, -1), dec_inputs)
    # pred = pred.data.max(1, keepdim=True)[1]
    # tgt_pre_len = 0
    # results = [idx2word[n.item()] for n in pred.squeeze()]  # 获取预测的结果
    # #print(results)
    # for i in range(len(results)):  # 因为E是一句话的结尾，所以取E之前的句子
    #     if results[i] == 'E':
    #         tgt_pre_len = i
    #         break
    # #print(tgt_pre_len)
    # #print(enc_inputs[0], '-->', [idx2word[n.item()] for n in pred.squeeze()])
    # print(enc_inputs[1], '-->', [results[i] for i in range(tgt_pre_len)])

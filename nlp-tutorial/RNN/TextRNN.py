# !/usr/bin/env python3
"""
TextRNN
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def make_batch():
    input_batch, target_batch = [], []

    for sen in sentences:
        word = sen.split()  # space tokenizer
        input = [word_dict[n] for n in word[:-1]]  # create (1~n-1) as input
        target = word_dict[word[-1]]  # create (n) as target, We usually call this 'casual language model'
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 使用确定性算法，但这可能会降低性能
    torch.backends.cudnn.deterministic = True
    # 禁用 cuDNN 的自动调优，这有助于减少不确定性
    torch.backends.cudnn.benchmark = False


class biRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(biRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 定义权重矩阵
        self.weight_ih = nn.ParameterList(
            [nn.Parameter(torch.Tensor(self.hidden_size, input_size)) for _ in range(num_layers)])
        self.weight_hh = nn.ParameterList(
            [nn.Parameter(torch.Tensor(self.hidden_size, hidden_size)) for _ in range(num_layers)])
        self.bias_ih = nn.ParameterList(
            [nn.Parameter(torch.Tensor(self.hidden_size)) for _ in range(num_layers)])
        self.bias_hh = nn.ParameterList(
            [nn.Parameter(torch.Tensor(self.hidden_size)) for _ in range(num_layers)])

        if self.bidirectional:
            self.weight_ih_reverse = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.hidden_size, input_size)) for _ in range(num_layers)])
            self.weight_hh_reverse = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.hidden_size, hidden_size)) for _ in range(num_layers)])
            self.bias_ih_reverse = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.hidden_size)) for _ in range(num_layers)])
            self.bias_hh_reverse = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.hidden_size)) for _ in range(num_layers)])

        # 初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for layer in range(self.num_layers):
            for weight in [self.weight_ih[layer], self.weight_hh[layer], self.bias_ih[layer], self.bias_hh[layer]]:
                nn.init.uniform_(weight, -stdv, stdv)
            if self.bidirectional:
                for weight in [self.weight_ih_reverse[layer], self.weight_hh_reverse[layer],
                               self.bias_ih_reverse[layer], self.bias_hh_reverse[layer]]:
                    nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input_data, h0=None):
        batch_size, seq_len, input_size = input_data.size()

        # 初始化隐藏状态
        if h0 is None:
            h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size,
                             device=input_data.device)
        hidden_states = []

        for layer in range(self.num_layers):
            layer_hidden_states = []
            h = h0[layer * self.num_directions:(layer + 1) * self.num_directions]

            if self.bidirectional:
                h_forward = h[0].clone()
                h_backward = h[1].clone()
            else:
                h_forward = h[0].clone()

            for t in range(seq_len):
                x = input_data[:, t, :]

                h_forward = torch.tanh(
                    torch.mm(x, self.weight_ih[layer].t()) + self.bias_ih[layer] + torch.mm(h_forward,
                                                                                            self.weight_hh[layer].t()) +
                    self.bias_hh[layer])
                layer_hidden_states.append(h_forward.unsqueeze(0))

                if self.bidirectional:
                    x_reverse = input_data[:, seq_len - t - 1, :]
                    h_backward = torch.tanh(
                        torch.mm(x_reverse, self.weight_ih_reverse[layer].t()) + self.bias_ih_reverse[layer] + torch.mm(
                            h_backward, self.weight_hh_reverse[layer].t()) + self.bias_hh_reverse[layer])
                    layer_hidden_states.append(h_backward.unsqueeze(0))

            if self.bidirectional:
                hidden_states.append(torch.cat(layer_hidden_states, dim=0))
            else:
                hidden_states.append(torch.stack(layer_hidden_states, dim=0))

            input_data = hidden_states[-1].transpose(0, 1)

        hidden_states = torch.cat(hidden_states, dim=0)
        return hidden_states[:, -1, :], h0


class TextRNN(nn.Module):

    def __init__(self):
        super(TextRNN, self).__init__()
        # self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.bi_rnn = biRNN(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, hidden, X):
        # pytorch定义的RNN
        # X = X.transpose(0, 1)  # 自定义RNN要注释调
        # outputs, hidden = self.rnn(X, hidden)
        # outputs = outputs[-1]

        # 自定义RNN
        outputs, hidden = self.bi_rnn(X, hidden)
        outputs = outputs[-1]
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        # [batch_size, num_directions(=1) * n_hidden]
        # model : [batch_size, n_class]
        model = self.W(outputs) + self.b
        return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_step = 2  # number of cells(= number of Step)
    n_hidden = 5  # number of hidden units in one cell
    num_layer = 1
    bidirectional = False
    num_directions = 2 if bidirectional else 1

    sentences = [
        "i like dog", "i love coffee", "i hate milk", "i love kitty", "i love you", "i hate him", "i trust you"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)
    batch_size = len(sentences)
    # 设置随机种子，保证每次运行结果一致
    set_seed(42)
    model = TextRNN().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch).to(device)
    target_batch = torch.LongTensor(target_batch).to(device)

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()

        # hidden : [num_layers * num_directions, batch, hidden_size]
        hidden = torch.zeros(num_layer * num_directions, batch_size, n_hidden).to(device)
        # input_batch : [batch_size, n_step, n_class]
        output = model(hidden, input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), ', loss: ', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    input = [sen.split()[:2] for sen in sentences]

    # Predict
    hidden = torch.zeros(1, batch_size, n_hidden).to(device)
    predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])

    # Test
    origin_test_text = "you love"
    test_text = [word_dict[n] for n in origin_test_text.split()]
    test_batch = [test_text]
    test_batch = np.eye(n_class)[test_batch]
    test_batch = torch.FloatTensor(test_batch).to(device)
    test_hidden = torch.zeros(num_layer * num_directions, 1, n_hidden).to(device)
    outputs = model(test_hidden, test_batch)
    # res = model(test_hidden, test_batch).data.max(1, keepdim=True)[1].squeeze()
    res = outputs.argmax(dim=1).cpu().squeeze()
    probabilities = F.softmax(outputs, dim=1).cpu()  # dim=1 表示在类别维度上应用softmax
    max_probs, _ = torch.max(probabilities, dim=1)
    print(origin_test_text, "->", number_dict[res.item()], ", probability: {}".format(max_probs[0].item()))

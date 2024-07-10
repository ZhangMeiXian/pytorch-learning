# !/usr/bin/env python3
"""
TextLSTM
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def make_batch():
    input_batch, target_batch = [], []

    for seq in seq_data:
        input_data = [word_dict[n] for n in seq[:-1]]  # 'm', 'a' , 'k' is input
        target_data = word_dict[seq[-1]]  # 'e' is target
        input_batch.append(np.eye(n_class)[input_data])
        target_batch.append(target_data)

    return input_batch, target_batch


# 自定义的LSTM比nn.LSTM慢很多，优化一下，实现计算加速

class OptimizedLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(OptimizedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combine all weight matrices into one for faster computation
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=True)

    def forward(self, x, h, c):
        # Ensure h is of the correct shape
        h = h.view(x.size(0), -1)
        combined = torch.cat((x, h), dim=1)
        gates = self.W(combined)

        i, f, g, o = torch.split(gates, self.hidden_size, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class OptimizedLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(OptimizedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm_cells = nn.ModuleList([
            OptimizedLSTMCell(input_size if i == 0 else hidden_size * self.num_directions, hidden_size)
            for i in range(num_layers)
        ])

        if self.bidirectional:
            self.lstm_cells_reverse = nn.ModuleList([
                OptimizedLSTMCell(input_size if i == 0 else hidden_size * self.num_directions, hidden_size)
                for i in range(num_layers)
            ])

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            h0 = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in
                  range(self.num_layers * self.num_directions)]
            c0 = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in
                  range(self.num_layers * self.num_directions)]
        else:
            h0, c0 = hidden
            h0 = [h0 for _ in range(self.num_layers * self.num_directions)]
            c0 = [c0 for _ in range(self.num_layers * self.num_directions)]

        outputs = []
        outputs_reverse = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h, c = h0[layer], c0[layer]
                h0[layer], c0[layer] = self.lstm_cells[layer](x_t, h, c)
                x_t = h0[layer]
            outputs.append(x_t.unsqueeze(1))

        if self.bidirectional:
            for t in range(seq_len - 1, -1, -1):
                x_t = x[:, t, :]
                for layer in range(self.num_layers):
                    h, c = h0[layer + self.num_layers], c0[layer + self.num_layers]
                    h0[layer + self.num_layers], c0[layer + self.num_layers] = self.lstm_cells_reverse[layer](x_t, h, c)
                    x_t = h0[layer + self.num_layers]
                outputs_reverse.append(x_t.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        if self.bidirectional:
            outputs_reverse = torch.cat(outputs_reverse[::-1], dim=1)
            outputs = torch.cat((outputs, outputs_reverse), dim=2)

        h_n = torch.stack(h0, dim=0)
        c_n = torch.stack(c0, dim=0)

        return outputs[:, -1, :], (h_n, c_n)

# 自定义LSTM实现


class LSTMCell(nn.Module):
    """
    LSTM细胞单元
    """

    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入门
        self.W_ii = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=True)

        # 遗忘门
        self.W_if = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=True)

        # 细胞门
        self.W_ig = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hg = nn.Linear(hidden_size, hidden_size, bias=True)

        # 输出门
        self.W_io = nn.Linear(input_size, hidden_size, bias=True)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, h, c):
        i = torch.sigmoid(self.W_ii(x) + self.W_hi(h))
        f = torch.sigmoid(self.W_if(x) + self.W_hf(h))
        g = torch.tanh(self.W_ig(x) + self.W_hg(h))
        o = torch.sigmoid(self.W_io(x) + self.W_ho(h))

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class LSTM(nn.Module):
    """
    自定义LSTM
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size * self.num_directions, hidden_size)
            for i in range(num_layers)
        ])

        if self.bidirectional:
            self.lstm_cells_reverse = nn.ModuleList([
                LSTMCell(input_size if i == 0 else hidden_size * self.num_directions, hidden_size)
                for i in range(num_layers)
            ])

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            h0 = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in
                  range(self.num_layers * self.num_directions)]
            c0 = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in
                  range(self.num_layers * self.num_directions)]
        else:
            h0, c0 = hidden
            h0 = [h0 for _ in range(self.num_layers * self.num_directions)]
            c0 = [c0 for _ in range(self.num_layers * self.num_directions)]

        outputs = []
        outputs_reverse = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h0[layer], c0[layer] = self.lstm_cells[layer](x_t, h0[layer], c0[layer])
                x_t = h0[layer]
            outputs.append(x_t.unsqueeze(1))

        if self.bidirectional:
            for t in range(seq_len - 1, -1, -1):
                x_t = x[:, t, :]
                for layer in range(self.num_layers):
                    h0[layer + self.num_layers], c0[layer + self.num_layers] = self.lstm_cells_reverse[layer](x_t, h0[
                        layer + self.num_layers], c0[layer + self.num_layers])
                    x_t = h0[layer + self.num_layers]
                outputs_reverse.append(x_t.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        if self.bidirectional:
            outputs_reverse = torch.cat(outputs_reverse[::-1], dim=1)
            outputs = torch.cat((outputs, outputs_reverse), dim=2)

        h_n = torch.stack(h0, dim=0)
        c_n = torch.stack(c0, dim=0)

        # 取最后一个时间步的输出作为最终输出
        return outputs[:, -1, :], (h_n, c_n)


class TextLSTM(nn.Module):

    def __init__(self):
        super(TextLSTM, self).__init__()
        # self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
        self.lstm = OptimizedLSTM(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        # input_data = X.transpose(0, 1)  # X : [n_step, batch_size, n_class], nn.LSTM需要转换
        hidden_state = torch.zeros(1, len(X), n_hidden).to(device)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state = torch.zeros(1, len(X), n_hidden).to(device)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(X, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.W(outputs) + self.b  # [batch_size, n_class]
        return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_step = 3  # number of cells(= number of Step)
    n_hidden = 128  # number of hidden units in one cell

    char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
    word_dict = {n: i for i, n in enumerate(char_arr)}
    number_dict = {i: w for i, w in enumerate(char_arr)}
    n_class = len(word_dict)  # number of class(=number of vocab)

    seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

    model = TextLSTM().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch).to(device)
    target_batch = torch.LongTensor(target_batch).to(device)

    # Training
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), ', loss: ', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()
    inputs = [sen[:3] for sen in seq_data]
    # predict = model(input_batch).data.max(1, keepdim=True)[1]
    outputs = model(input_batch)
    predict = outputs.argmax(dim=1).cpu()
    print(inputs, '->', [number_dict[n.item()] for n in predict.squeeze()])
    # new test
    test_txt = "fin"
    test_idx = [word_dict[c] for c in test_txt]
    test_batch = [np.eye(n_class)[test_idx]]
    test_batch = torch.FloatTensor(test_batch).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(test_batch)
    probabilities = F.softmax(outputs, dim=1).cpu()
    max_probs, _ = torch.max(probabilities, dim=1)
    predict = outputs.argmax(dim=1).cpu()
    print(test_txt, "->", number_dict[predict.item()], ", probability: ", max_probs.item())
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
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        input = [word_dict[n] for n in words[:(i + 1)]]
        input = input + [0] * (max_len - len(input))
        target = word_dict[words[i + 1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        self.W = nn.Linear(n_hidden * 2, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        input = X.transpose(0, 1)  # input : [n_step, batch_size, n_class]

        hidden_state = torch.zeros(1*2, len(X), n_hidden).to(device)   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.zeros(1*2, len(X), n_hidden).to(device)     # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model


if __name__ == '__main__':
    # 预测当前面这些词出现时，下一个词最大概率是什么
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_hidden = 5 # number of hidden units in one cell

    sentence = (
        'Lorem ipsum dolor sit amet consectetur adipisicing elit '
        'sed do eiusmod tempor incididunt ut labore et dolore magna '
        'aliqua Ut enim ad minim veniam quis nostrud exercitation'
    )

    word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
    number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}
    n_class = len(word_dict)
    max_len = len(sentence.split())

    model = BiLSTM().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch).to(device)
    target_batch = torch.LongTensor(target_batch).to(device)

    # Training
    for epoch in range(10000):
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), ', loss: ', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(sentence)
    print([number_dict[n.item()] for n in predict.squeeze()])
    # new Test：调换单词顺序看看会预测出什么吧
    test_txt = "aliqua ipsum"
    test_txt = test_txt.split()
    text_idx = [word_dict[word] for word in test_txt]
    text_idx = text_idx + [0] * (max_len - len(text_idx))
    test_batch = [np.eye(n_class)[text_idx]]
    test_batch = torch.FloatTensor(test_batch).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(test_batch)
    probs = F.softmax(outputs, dim=1).cpu()
    max_prob, _ = torch.max(probs, dim=1)
    predict = outputs.argmax(dim=1).cpu()
    print(test_txt, "->", number_dict[predict.item()], "probability: {}".format(max_prob[0].item()))
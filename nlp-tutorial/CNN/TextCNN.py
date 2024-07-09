# !/usr/bin/env python3
"""
TextCNN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    TextCNN
    """

    def __init__(self):
        super(TextCNN, self).__init__()
        self.num_filters_total = num_filters * len(filter_sizes)
        self.W = nn.Embedding(vocab_size, embedding_size)
        self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
        self.Bias = nn.Parameter(torch.zeros([num_classes]))
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])

    def forward(self, X):
        embedded_chars = self.W(X)  # [batch_size, sequence_length, embedding_size]
        # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        embedded_chars = embedded_chars.unsqueeze(1)
        pooled_outputs = []

        for i, conv in enumerate(self.filter_list):
            # 采用不同的卷积核对时序embeeding进行处理，获取不同的特征向量
            # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
            h = F.relu(conv(embedded_chars))
            # mp : ((filter_height, filter_width))
            mp = nn.MaxPool2d((sequence_length - filter_sizes[i] + 1, 1))
            # pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)

        # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        h_pool = torch.cat(pooled_outputs, len(filter_sizes))
        # [batch_size(=6), output_height * output_width * (output_channel * 3)]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total])
        # [batch_size, num_classes]
        model = self.Weight(h_pool_flat) + self.Bias
        return model


if __name__ == '__main__':
    embedding_size = 2  # embedding size
    sequence_length = 3  # sequence length
    num_classes = 2  # number of classes
    filter_sizes = [2, 2, 2]  # n-gram windows
    num_filters = 3  # number of filters

    # 3 words sentences (=sequence_length is 3)
    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences]).to(device)
    targets = torch.LongTensor([out for out in labels]).to(device)  # To using Torch Softmax Loss function

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(inputs)

        # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, targets)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), ', loss: ', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # 测试训练的模型
    test_text = 'sorry hate you'
    tests = [np.asarray([word_dict[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(tests).to(device)

    # Predict
    # predict = model(test_batch).data.max(1, keepdim=True)[1]
    outputs = model(test_batch)
    probabilities = F.softmax(outputs, dim=1).cpu()  # dim=1 表示在类别维度上应用softmax
    max_probs, _ = torch.max(probabilities, dim=1)
    predict = outputs.argmax(dim=1).cpu()
    print("probabilities: {}, predict: {}".format(max_probs[0].item(), predict.item()))
    if predict.item() == 0:
        print(test_text, "is Bad Mean...")
    else:
        print(test_text, "is Good Mean!!")
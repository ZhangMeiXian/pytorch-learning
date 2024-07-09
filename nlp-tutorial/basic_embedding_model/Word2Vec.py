# !/usr/bin/env python3
"""
Word2Vec
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def random_batch():
    """
    随机分批次
    :return:
    """
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)
    for i in random_index:
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])  # target
        random_labels.append(skip_grams[i][1])   # context word
    return random_inputs, random_labels


class Word2Vec(nn.Module):
    """
    Word2Vec模型
    """

    def __init__(self):
        super(Word2Vec, self).__init__()
        # W and WT is not Traspose relationship
        self.W = nn.Linear(voc_size, embedding_size, bias=False)  # voc_size > embedding_size Weight
        self.WT = nn.Linear(embedding_size, voc_size, bias=False)  # embedding_size > voc_size Weight

    def forward(self, X):
        # X : [batch_size, voc_size]
        hidden_layer = self.W(X)  # hidden_layer : [batch_size, embedding_size]
        # 将嵌入向量转换回词汇表的概率分布
        output_layer = self.WT(hidden_layer)  # output_layer : [batch_size, voc_size]
        return output_layer


def get_word_vector(model, word, word_dict):
    """
    获取单词的向量表示
    """
    word_idx = word_dict[word]
    one_hot = torch.zeros(1, voc_size)
    one_hot[0][word_idx] = 1
    model.eval()
    with torch.no_grad():
        hidden_layer = model.W(one_hot)
    return hidden_layer


def get_sentence_vector(model, sentence, word_dict):
    """
    获取句子的向量表示
    """
    vectors = [get_word_vector(model, word, word_dict) for word in sentence.split()]
    sentence_vector = torch.stack(vectors).mean(dim=0)
    return sentence_vector


if __name__ == '__main__':
    batch_size = 2  # mini-batch size
    embedding_size = 2  # embedding size

    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]
    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    voc_size = len(word_list)
    # Make skip gram of one size window
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
        for w in context:
            skip_grams.append([target, w])

    model = Word2Vec()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []
    for epoch in range(5000):
        input_batch, target_batch = random_batch()
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)
        optimizer.zero_grad()
        output = model(input_batch)
        # output : [batch_size, voc_size], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), ', loss: ', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # # 绘制不同的单词的分布
    # for i, label in enumerate(word_list):
    #     W, WT = model.parameters()
    #     x, y = W[0][i].item(), W[1][i].item()
    #     plt.scatter(x, y)
    #     plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    # # plt.show()
    # plt.savefig('./word2vec.png')

    # 采用训练好的模型进行向量化表征
    cur_word = 'dog'
    print(get_word_vector(model, cur_word, word_dict))
    sentences = "dog apple cat"
    print(get_sentence_vector(model, sentences, word_dict))
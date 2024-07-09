# !/usr/bin/env python3
"""
Predict Next Word with NNLM
简单的下一句预测模式
paper link: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
"""
import torch
import torch.nn as nn
import torch.optim as optim


def make_batch():
    """

    :return:
    """
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input_list = [word_dict[n] for n in word[:-1]]
        target_list = word_dict[word[-1]]

        input_batch.append(input_list)
        target_batch.append(target_list)
    return input_batch, target_batch


class NNLM(nn.Module):

    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        self.W = nn.Linear(n_step * m, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, X):
        # 转换成索引的文本向量化
        X = self.C(X)  # X : [batch_size, n_step, m]
        X = X.view(-1, n_step * m)  # [batch_size, n_step * m]
        tanh = torch.tanh(self.d + self.H(X))  # [batch_size, n_hidden]
        output = self.b + self.W(X) + self.U(tanh)  # [batch_size, n_class]
        return output


if __name__ == '__main__':
    # 预测下一个单词的前缀长度
    n_step = 3  # number of steps, n-1 in paper
    # 隐藏层的维度
    n_hidden = 2  # number of hidden size, h in paper
    # 向量化的词向量维度
    m = 2  # embedding size, m in paper

    # 固定长度的序列
    sentences = ["i don't like dog", "i really love coffee", "you may hate milk", "you might like me"]
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    # 文本到索引的字典映射
    word_dict = {w: i for i, w in enumerate(word_list)}
    # 索引到文本的字典映射
    number_dict = {i: w for i, w in enumerate(word_list)}
    # 总词汇量
    n_class = len(word_dict)  # number of Vocabulary

    model = NNLM()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), '， loss:', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    predict = model(input_batch).data.max(1, keepdim=True)[1]

    print([sen.split()[:3] for sen in sentences], "->", [number_dict[n.item()] for n in predict.squeeze()])
    # [['i', "don't", 'like'], ['i', 'really', 'love'], ['you', 'may', 'hate'], ['you', 'might', 'like']] -> ['dog', 'coffee', 'milk', 'me']

    # 尝试采用新的组合进行预测
    new_sentence = "i might like"
    words = new_sentence.split()
    input_ids = [word_dict[word] for word in words]
    input_tensor = torch.LongTensor(input_ids)
    # 进行预测
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted = torch.max(output, 1)
    predicted_word = number_dict[predicted.item()]
    print("{} -> {}".format(new_sentence, predicted_word))
    # i might like -> me
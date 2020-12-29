import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")
# print(data.tail(10))

words = list(set(data["Word"].values))
n_words = len(words)

# print(n_words, "\n")


class SentenceGetter(object):
    def __init__(self, data):
        self.data = data
        self.n_sent = 1
        self.empty = False

    def get_next(self):
        try:
            s = self.data[self.data["Sentence #"] == "Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s["Word"].values.tolist(), s["POS"].values.tolist(), s["Tag"].values.tolist()
        except:
            self.empty = True
            return None, None, None


getter = SentenceGetter(data)
sent, pos, tag = getter.get_next()
print(sent, pos, tag)
for w, t in zip(sent, tag):
    print(w, t)


class MemoryTagger(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        voc = {}  # 词表 注意看 voc 的存储形式。
        self.tags = []  # 标签
        for w, t in zip(X, y):    # 这里是统计每个单词作为各种标签的频率。
            if t not in self.tags:   # tags用来记录这句话中标签的种类的
                self.tags.append(t)
            if w not in voc:     # voc里面没有这个单词， 就添加这个单词
                voc[w] = {}
            if t not in voc[w]:
                # 这个单词下面没有这个标签，那么就给这个单词的记录标签频率的字典里面添加这个标签
                voc[w][t] = 0
            voc[w][t] += 1  # 该单词对应的标签频率+1

        # 所以最后voc的存储形式： voc = {w1: {tag1:f1, tag2:f2...},w2: {tag1:f1, tag2:f2...},w3: {tag1:f1, tag2:f2...}...}

        self.memory = {}
        for k, d in voc.items():
            self.memory[k] = max(d, key=d.get)  # 取最大的频率对应的标签作为单词的预测值。
        print("this is memory".format(), self.memory)

    def predict(self, X, y=None):  # X 是句子列表
        return[self.memory.get(x, "O") for x in X]  # get x， 默认是"O"


# 测试一下
tagger = MemoryTagger()
tagger.fit(sent, tag)
print(tagger.predict(sent))

# 用所有数据进行训练
words = data["Word"].values.tolist()
print(len(words))
tags = data["Tag"].values.tolist()
pred = cross_val_predict(estimator=MemoryTagger(), X=words, y=tags, cv=5)
report = classification_report(y_pred=pred, y_true=tags)
print(report, "\n")

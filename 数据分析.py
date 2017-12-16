from recognize import load_data
import numpy as np
import collections
import matplotlib.pyplot as plt

"""
分析生成的数据的特征
"""


def see_distribution(data):
    labels = [i['ans'] for i in data]
    print("类别总数", len(set(labels)))
    train = collections.Counter(labels)
    for i in sorted(train):
        print(i, train[i])


data = load_data()
see_distribution(data)
sha = np.array([i['data'].shape for i in data])
ma = np.max(sha, axis=0)
print(ma)
h = collections.Counter(sha[:, 0])
w = collections.Counter(sha[:, 1])
w = np.array([(i, w[i]) for i in sorted(w.keys())])
h = np.array([(i, h[i]) for i in sorted(h.keys())])
print('宽度分布', w)
print('高度分布', h)
plt.subplot("121")
plt.bar(h[:, 0], h[:, 1])
plt.subplot("122")
plt.bar(w[:, 0], w[:, 1])
plt.show()
for i in data:
    if i['data'].shape[1] > 21:
        plt.title(i['data'].shape)
        plt.imshow(i['data'], cmap="gray")
        plt.show()

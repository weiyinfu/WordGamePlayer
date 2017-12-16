"""
使用卷积神经网络进行图片识别
"""
import collections
import bidict
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from recognize import load_data

image_rows = 22
image_cols = 30
label_number = bidict.bidict()


def uni_size(data):
    # 将图片统一大小
    sha = np.array([i.shape for i in data])
    if np.any(sha > (image_rows, image_cols)):
        raise Exception("以下图片尺寸大于最大尺寸：%s" % ("\n".join(i['filename'] for i in data)))
    a = np.zeros((len(data), image_rows, image_cols))
    for i in range(len(data)):
        pos_x, pos_y = (np.array([image_rows, image_cols]) - data[i].shape) // 2
        a[i, pos_x:pos_x + data[i].shape[0], pos_y:data[i].shape[1] + pos_y] = data[i]
    return a


def get_data():
    data = load_data()
    for i in data:
        if i['ans'] not in label_number:
            label_number[i['ans']] = len(label_number)
        i['y'] = label_number.get(i['ans'])
    x = uni_size([i['data'] for i in data])
    y = np.array([i['y'] for i in data])
    return x, y


def see_data():
    x, y = get_data()
    for i in x:
        plt.imshow(i)
        plt.show()


def get_label(cls_number):
    return label_number.inv.get(cls_number)


class KNN:
    def __init__(self):
        x, y = get_data()
        x = x.reshape(len(x), -1)
        pca = PCA(n_components=16)
        x = x.reshape(len(x), -1)
        x = pca.fit_transform(x)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(x, y)
        self.pca = pca
        self.clf = clf

    def predict(self, recs):
        recs = [i for i in recs if all(np.array(i['data'].shape) < (image_rows, image_cols))]
        data = uni_size([i['data'] for i in recs])
        data = data.reshape(-1, image_rows * image_cols)
        data = self.pca.transform(data)
        y_mine = self.clf.predict(data)
        for i in range(len(recs)):
            recs[i]['ans'] = get_label(y_mine[i])
        return recs

    def predict_ndarray(self, data):
        data = data.reshape(-1, image_rows * image_cols)
        data = self.pca.transform(data)
        return self.clf.predict(data)


class RandomForest:
    def __init__(self):
        x, y = get_data()
        clf = RandomForestClassifier()
        x = x.reshape(len(x), -1)
        clf.fit(x, y)
        self.clf = clf

    def predict(self, recs):
        recs = [i for i in recs if all(np.array(i['data'].shape) < (image_rows, image_cols))]
        data = uni_size([i['data'] for i in recs])
        data = data.reshape(-1, image_rows * image_cols)
        y_mine = self.clf.predict(data)
        for i in range(len(recs)):
            recs[i]['ans'] = get_label(y_mine[i])
        return recs

    def predict_ndarray(self, data):
        data = data.reshape(-1, image_rows * image_cols)
        return self.clf.predict(data)


class DecisionTree:
    def __init__(self):
        x, y = get_data()
        clf = tree.DecisionTreeClassifier()
        x = x.reshape(len(x), -1)
        clf.fit(x, y)
        self.clf = clf

    def predict(self, recs):
        recs = [i for i in recs if all(np.array(i['data'].shape) < (image_rows, image_cols))]
        data = uni_size([i['data'] for i in recs])
        y_mine = self.clf.predict(data)
        for i in range(len(recs)):
            recs[i]['ans'] = get_label(y_mine[i])
        return recs

    def predict_ndarray(self, data):
        data = data.reshape(-1, image_rows * image_cols)
        return self.clf.predict(data)


def test_clf(debug=True):
    """
    测试分类器正确率
    :param debug:
    :return:
    """
    x, y = get_data()
    eye = KNN()
    yy = eye.predict_ndarray(x.reshape(len(x), -1))
    data = np.vstack((y, yy)).T
    print(np.count_nonzero(y == yy), len(y), np.count_nonzero(y == yy) / len(y))
    if debug:
        for i in range(len(y)):
            if y[i] != yy[i]:
                plt.imshow(x[i])
                plt.title("true:" + get_label(y[i]) + " mine:" + get_label(yy[i]))
                plt.show()


if __name__ == '__main__':
    test_clf(debug=False)

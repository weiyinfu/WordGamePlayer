import numpy as np
import collections
from sklearn.cluster import KMeans
from recognize import log

"""
本模块对小矩形进行排序
"""
english_dic = set()
with open("words.txt") as f:
    for line in f:
        if line:
            english_dic.add(line.split()[1])


def sort_words(words):
    # 对单词进行排序，防止单词覆盖
    # 优先拼写较长的单词，因为较长的单词表明该单词没有被覆盖
    log.info("enter")
    v = [0] * len(words)
    for ind, i in enumerate(words):
        w = ''.join(map(lambda x: x['ans'], i))
        print(w)
        if w in english_dic:
            v[ind] = len(w)
    v = np.argsort(v)
    a = [] 
    for i in range(len(v) - 1, -1, -1):
        a.append( words[v[i]])
    log.info("end")
    return a


def intersect(m, n):
    # 判断两个矩形框高度相交
    if m['max_x'] < n['min_x']:
        return False
    if m['min_x'] > n['max_x']:
        return False
    return True


def build_words(recs):
    # 根据小矩形构建单词
    log.info("enter")
    char_gap_width = 9

    recs = sorted(recs, key=lambda x: x['min_y'])
    a = []
    for i in recs:
        had = False
        for ind, word in enumerate(a):
            if word[-1]['max_y'] + char_gap_width > i['min_y'] and intersect(word[-1], i):
                word.append(i)
                had = True
        if not had:
            a.append([i])
    log.info("end")
    return a


def find_frequent(a):
    """
    给定一个数组，求数组中的众数
    方法是对元素进行KMEANS聚类，聚成3类，取元素最多的那一类
    这种方法可以寻找近似众数
    :param a:
    :return:
    """
    log.info("enter")
    a = np.array(a)
    km = KMeans(n_clusters=min(len(a), 3))
    label = km.fit_predict(np.reshape(a, (-1, 1)))
    cnt = collections.Counter(label)
    ma = None
    for i in cnt:
        if ma is None or cnt[ma] < cnt[i]:
            ma = i
    baseline = np.mean(a[label == ma])
    log.info("end")
    return baseline


def check_baseline(a):
    # 检查每个单词的baseline，如果baseline不同，则分离出去
    # 字符的高度有两种，一等高度和二等高度
    log.info("enter")
    ans = []
    for i in a:
        baseline = []
        for ch in i:
            h = ch['max_x'] - ch['min_x']
            if abs(h - 15) < 3:  # 只统计占一格的字符
                baseline.append(ch['max_x'])
        if len(baseline) < 3:
            ans.append(i)
            continue
        baseline_value = find_frequent(baseline)
        new_word = []
        for ch in i:
            h = ch['max_x'] - ch['min_x']
            if abs(h - 15) < 3:  # 如果占一格
                if abs(ch['max_x'] - baseline_value) > 4:
                    ans.append([ch])  # 占一格且远离基线，必然不属于当前word
                else:
                    new_word.append(ch)
            else:
                new_word.append(ch)
        ans.append(new_word)
    log.info("end")
    return ans


def sort_recs(recs):
    """
    第二种排序方法
    从左到右扫描，如果能归入到现有词汇里面就归入，否则新创建一个
    :param recs:
    :return:
    """
    a = build_words(recs)
    for i in a:
        print("".join(map(lambda x: x['ans'], i)), end=' ')
    print()
    # a = check_baseline(a)#如果保证精度就会丧失速度
    a = sort_words(a)
    ret = []
    for i in a:
        ret.extend(i)
    return ret

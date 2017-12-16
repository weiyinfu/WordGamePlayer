from recognize import *
import warnings

"""
分析hand-label文件夹中的图片，将结果输出到data文件夹
"""

warnings.filterwarnings("ignore")

hand_label = "hand-label"
target = "data"


def parse_label(s):
    if s[-4:].lower() == '.jpg':
        s = s[:-4]
    if '-' in s:
        s = s[:s.index("-")]
    a = []
    i = 0
    while i < len(s):
        if s[i] == '(':
            now = ''
            for j in range(i + 1, len(s)):
                if s[j] == ')': break
                now += s[j]
            i = j + 1
            a.append(now)
        else:
            a.append(s[i])
            i += 1
    return a


def go(force=False):
    """
    将手工标注的数据生成为单个的字符图片
    :param force:  是否强制生成（不论文件新旧）
    :return:
    """

    def getpath(y, file_id):
        return os.path.join(target, "%s-%d.jpg" % (y, file_id))

    # 如果目标文件夹比手标文件夹新，那就不更新，所以不要手动更新data文件夹
    # 那样会导致无法由hand_label更新
    if not force and os.path.getmtime(target) > os.path.getmtime(hand_label):
        return
    for i in os.listdir(target):
        os.remove(os.path.join(target, i))
    for i in os.listdir(hand_label):
        img, rec = img2recs(os.path.join(hand_label, i))
        rec = sorted(rec, key=lambda x: x['min_y'])
        ans = parse_label(i)
        if len(rec) != len(ans):
            raise Exception('训练数据有问题，{}文件有{}个字符，而我发现了{}个字符'.format(os.path.join(hand_label, i), len(i) - 4, len(rec)))
        for y, x in zip(ans, rec):
            file_id = 0
            while os.path.exists(getpath(y, file_id)):
                file_id += 1
            io.imsave(getpath(y, file_id), x['data'])


if __name__ == '__main__':
    go(force=True)

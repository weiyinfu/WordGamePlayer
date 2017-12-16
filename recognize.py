import logging
import os

import numpy as np
from skimage import io, color, img_as_float
import matplotlib.pyplot as plt
"""
本模块是识别图像最重要的部分
"""
# 使用日志发现系统瓶颈
log = logging.getLogger()
log_handler = logging.FileHandler("haha.log", 'w', 'utf8')
log_handler.setFormatter(logging.Formatter("%(asctime)s $ %(funcName)s $ %(message)s"))
log.addHandler(log_handler)
log.setLevel("INFO")

GREY_COLOR = 128 / 256
WHITE_COLOR = 255 / 256


def groups(data, ceil_cnt=400, floor_cnt=15):
    """
    执行图像中像素聚类
    :param data: 灰度图片
    :param ceil_cnt: 每个聚类的上限
    :param floor_cnt: 每个聚类大小的下限
    :return: number，cnt都是二维数组，number表示每个像素的id，cnt表示每个像素的个数
    一个隐秘的bug，如果字符相连，导致聚类过大，导致直接删掉该聚类
    """
    # 聚类O(n*n)复杂度
    log.info('enter')
    a = data.reshape(-1)
    number = np.arange(len(a))  # 每个像素点的id
    cnt = np.ones(len(a))  # 每个聚类包含的像素个数

    def findfather(x):
        # 并查集
        if number[x] == x:
            return x
        number[x] = findfather(number[x])
        return number[x]

    def legal(x, y):
        return 0 <= x < data.shape[0] and 0 <= y < data.shape[1]

    valid_pos = np.argwhere(a)  # 只处理有颜色的地方
    for i in valid_pos:  # 使用一维循环代替二维循环
        # 对上下左右四个方向进行循环
        for di in (-data.shape[1], -1, -data.shape[1] - 1, -data.shape[1] + 1):
            j = i + di
            if len(a) > j >= 0 and a[j]:
                fa1 = findfather(i)
                fa2 = findfather(j)
                if fa1 != fa2:  # 分支合并
                    number[fa1] = fa2
                    cnt[fa2] += cnt[fa1]
    for i in valid_pos:
        number[i] = findfather(i)
        cnt[i] = cnt[number[i]]
    # 将聚类个数不符合的聚类清除掉
    a[np.logical_or(cnt < floor_cnt, cnt > ceil_cnt)] = 0
    log.info("end")
    return number, cnt


def to_grey(img, savefile=False):
    log.info("enter")
    # 在HSV空间中比较容易找出黄色
    hsv = color.rgb2hsv(img)
    bright = hsv[:, :, 2] > 0.8
    yellow = np.logical_and(hsv[:, :, 0] > 0.15, hsv[:, :, 0] < 0.2)
    np.logical_and(yellow, bright, out=yellow)  # 去掉灰暗部分
    img[np.logical_not(yellow)] = [0, 0, 0]  # 非黄色置为黑色
    if savefile:
        io.imsave("去掉非黄色像素.jpg", img)
    data = color.rgb2gray(img)
    log.info("exit")
    return data


def img2recs(filepath=None, img=None, savefile=False):
    """
    处理图片
    :param filepath: 待处理的图片路径
    :param img: 待处理的图片
    :return: 处理之后的图片，各个聚类
    """
    log.info('enter')
    if filepath:
        img = io.imread(filepath)
    if len(img.shape) == 3:  # 若为RGB图，则灰度化之
        img = to_grey(img, savefile)
    # 灰度化之后再次去掉非黄色像素
    img[np.abs(img - 220 / 256) > 40 / 256] = 0
    # 将图像二值化
    img[img > GREY_COLOR] = WHITE_COLOR
    img[img < GREY_COLOR] = 0
    number, cnt = groups(img)  # 每个元素所属类别，每个类别包含的元素个数
    rec = dict()
    # 求出每个聚类所在的外包矩形
    for i, j in np.argwhere(img):
        # h获取i,j点所在聚类的id
        n = number[i * img.shape[1] + j]
        if n not in rec:
            rec[n] = {'min_x': 0xfffff, 'min_y': 0xfffff,
                      'max_x': 0, 'max_y': 0,
                      'cnt': cnt[n]  # 属于该类别的个数
                      }
        rec[n]['min_x'] = min(rec[n]['min_x'], i)
        rec[n]['min_y'] = min(rec[n]['min_y'], j)
        rec[n]['max_x'] = max(rec[n]['max_x'], i)
        rec[n]['max_y'] = max(rec[n]['max_y'], j)
    for i, v in rec.items():
        v['data'] = img[v['min_x']:v['max_x'] + 1, v['min_y']:v['max_y'] + 1]
    if savefile:
        io.imsave("灰度化之后.jpg", img)
    log.info("end")
    return img, list(rec.values())


def load_data():
    # 读取数据
    templates = []
    path = "data"
    for i in os.listdir(path):
        tem = io.imread(os.path.join(path, i))
        tem = img_as_float(tem)
        tem[tem > GREY_COLOR] = WHITE_COLOR
        tem[tem < GREY_COLOR] = 0
        filename = os.path.basename(i)
        ans = filename[:filename.rindex(".")] if '-' not in filename else filename[:filename.index('-')]
        templates.append({'ans': ans, 'data': tem, 'filename': i})
    return templates

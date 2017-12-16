from PIL import Image
import numpy as np
import timeit

"""
PIL中过滤颜色有point函数，
这个函数跟numpy谁快呢？
"""
w = 1000
h = 1000


def getimg():
    a = np.random.randint(0, 255, (h, w, 3)).astype(np.byte)
    img = Image.fromarray(a, "RGB")
    return img


def pil(img):
    def filt(rgb):  # 把255种颜色进行过滤，这种过滤是过滤每一个通道
        if np.var(np.array(rgb)) < 10:
            return [0, 0, 0]
        else:
            return rgb

    img.point(filt)
    a = np.array(img.getdata())


def testNp(a):
    a[np.var(a, axis=2) < 10] = [0, 0, 0]


one = getimg()
res = timeit.timeit(lambda: pil(one), number=1)
print(res)
img = getimg()
a = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
res = timeit.timeit(lambda: testNp(a), number=1)
print(res)

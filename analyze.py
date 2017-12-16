import matplotlib.pyplot as plt

from recognize import *
from 生成数据 import go
from 传统机器学习方法 import KNN, get_label, uni_size
from 排序器 import sort_recs

"""
用于检测查看grab中的截图，分析问题
"""
go()
for i in os.listdir("tmp"):
    os.remove(os.path.join('tmp', i))
img, res = img2recs("grab/138.jpg", savefile=True)
res = sorted(res, key=lambda x: (x['min_x'], x['min_y']))
clf = KNN()
recs = []
for j, i in enumerate(res):
    io.imsave("tmp/%d.jpg" % j, i['data'])
    x = uni_size([i['data']])
    y = clf.predict_ndarray(x)[0]
    y = get_label(y)
    if y:
        i['ans'] = y
        recs.append(i)
recs = sort_recs(recs)
ans = []
show_img = False
for ch in recs:
    ans.append(ch['ans'])
    if show_img:
        print(ch['ans'], ch['min_x'], ch['min_y'], ch['max_x'], ch['max_y'])
        plt.title("min_x:%d,min_y:%d,max_x:%d,max_y:%d" % (ch['min_x'], ch['min_y'], ch['max_x'], ch['max_y']))
        plt.imshow(ch['data'], cmap="gray")
        plt.show()
print(ans)

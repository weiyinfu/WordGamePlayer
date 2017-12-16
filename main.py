import time

from pykeyboard import PyKeyboard
from skimage import io
from recognize import img2recs, log
from 传统机器学习方法 import KNN
from 截图 import grab
from 排序器 import sort_recs

# eye = DecisionTree()
eye = KNN()
# eye = RandomForest()
# from cnn import CNN
#
# eye = CNN()
imgId = 0
save_grab = True


def go(img):
    log.info("enter")
    img, recs = img2recs(img=img)
    if len(recs) == 0: return
    recs = eye.predict(recs)
    if len(recs) == 0: return
    recs = sort_recs(recs)
    ans = "".join(map(lambda i: i['ans'], recs))
    print(imgId, ans)
    log.info("end")
    return ans


def grab_img():
    # 获取屏幕截图
    global imgId
    log.info('enter')
    # 图片越大截图花费时间越多，所以截图应该尽量小
    # img = ImageGrab.grab((250, 161, 1141, 610))
    filename = "grab/%d.jpg" % imgId
    grab(filename, 62, 160, 857, 575)
    imgId += 1
    img = io.imread(filename)
    log.info("end")
    return img


print("open the window now, game will start in 5 seconds......")
time.sleep(5)
print("I am ready")
k = PyKeyboard()
while 1:
    img = grab_img()
    ans = go(img)
    if ans:
        k.type_string(ans)
        time.sleep(0.3)

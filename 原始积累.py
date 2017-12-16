from PIL import ImageGrab
import time
"""
此程序用于定时截图，可以用来积累最开始的数据
"""
cnt = 1
while 1:
    img = ImageGrab.grab((62, 160, 857,575))
    img.save("grab/{}.jpg".format(cnt))
    print(cnt)
    cnt += 1
    time.sleep(1)

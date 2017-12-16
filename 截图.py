import win32api
import win32con
import win32gui
import win32ui
"""
使用windows API进行截图
"""

def grab(filename, left, top, right, bottom):
    hwnd = 0  # 窗口的编号
    # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
    hwndDC = win32gui.GetWindowDC(hwnd)
    # 根据窗口的DC获取mfcDC
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    # mfcDC创建可兼容的DC
    saveDC = mfcDC.CreateCompatibleDC()
    # 创建bigmap准备保存图片
    saveBitMap = win32ui.CreateBitmap()
    # 获取监控器信息
    MoniterDev = win32api.EnumDisplayMonitors(None, None)
    w = MoniterDev[0][2][2]
    h = MoniterDev[0][2][3]
    right = min(w, right)
    bottom = min(h, bottom)
    # print w,h　　　#图片大小
    # 为bitmap开辟空间
    saveBitMap.CreateCompatibleBitmap(mfcDC, right - left, bottom - top)
    # 高度saveDC，将截图保存到saveBitmap中
    saveDC.SelectObject(saveBitMap)
    # 截取从左上角（0，0）长宽为（w，h）的图片
    saveDC.BitBlt((0, 0), (right - left, bottom - top), mfcDC, (left, top), win32con.SRCCOPY)
    saveBitMap.SaveBitmapFile(saveDC, filename)


if __name__ == '__main__':
    grab("grab/1.jpg", 100, 100, 200, 200)

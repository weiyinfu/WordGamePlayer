# 除了使用opencv，还可以使用VideoCapture这个库
import VideoCapture
import tkinter

root = tkinter.Tk()
dev = VideoCapture.Device(0, 1)
dev.getBuffer()
root.mainloop()

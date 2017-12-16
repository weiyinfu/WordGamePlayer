from PIL import Image, ImageTk
import tkinter as tk

img = Image.open("../haha.png").convert('L')
w, h = img.size
root = tk.Tk()
ca = tk.Canvas(width=w, height=h)
ca.pack()
tkimg = ImageTk.PhotoImage(img)
res = ca.create_image(w / 2, h / 2, image=tkimg)


def move(event):
    res = img.getpixel((event.x, event.y))
    root.title(res)


ca.bind("<Motion>", move)
root.mainloop()

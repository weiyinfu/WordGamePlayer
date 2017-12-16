import os

for i in "tmp data grab model".split():
    if not os.path.exists(i):
        os.mkdir(i)

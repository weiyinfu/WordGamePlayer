import time
import matplotlib.pyplot as plt
import numpy as np

"""
统计日志，发现瓶颈
"""
a = [i.split("$") for i in open("haha.log").readlines() if i.strip()]
use = dict()
for i in a:
    for j in range(len(i)):
        i[j] = i[j].strip()
    ms = int(i[0][i[0].rindex(',') + 1:]) / 1000
    i[0] = time.mktime(time.strptime(i[0], "%Y-%m-%d %H:%M:%S,%f")) + ms
    t, func, event = i
    if ':' in event:
        event, spot = event.split(":")
        event = event.strip()
        spot = spot.strip()
        func = func + ":" + spot
    if func not in use:
        use[func] = {
            'enter': 0,
            'sum': 0,
            'cnt': 0
        }
    if event in ('enter', 'spotEnter'):
        use[func]['enter'] = t
    elif event in ('end', 'spotEnd'):
        use[func]['cnt'] += 1
        use[func]['sum'] += t - use[func]['enter']

x = use.keys()
y = [i['sum'] / max(1, i['cnt']) for i in use.values()]
plt.bar(np.arange(len(x)), y, width=1, alpha=0.3, color='r', label='Func-Time')
plt.title("The cost of time of each function")
plt.xlabel("function")
plt.ylabel("time(second)")
plt.xticks(np.arange(len(x)) + 0.5, x)
plt.tight_layout()
plt.show()
print(use)

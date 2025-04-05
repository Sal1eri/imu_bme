import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


file = 'data.txt'
data = []

with open(file, 'r', encoding='utf-8') as file:
    for line in file:
        # 去除行末的换行符（\n）并添加到列表
        data.append(line.strip())

data = [int(i) for i in data]
# data = np.array(data)/data[0]
data = data[:110]
print(data)
plt.figure(figsize=(10,5))
plt.plot(data, 'b-',linewidth=2,marker='o',markersize=4)
plt.title('类器官成长曲线')
plt.xlabel('Times(hours)')
plt.ylabel('Area(pixels)')
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('data_curves.png')
plt.show()
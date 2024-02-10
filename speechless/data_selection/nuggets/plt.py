import matplotlib.pyplot as plt
import json
# import numpy as np

values = []
with open("/home/liyunshui/LLMs/code/instruct-thinking/datasets/dumped/alpaca_data_sorted_score_sample.json", "r") as f:
    data = json.load(f)
    for e in data:
        values.append(e["score"])
# 数据
values.reverse()
x = range(len(values))
plt.figure(figsize=(9, 6))  # 调整图形大小
# x_min = -5000  # 最小值
# x_max = 54000  # 最大值
# plt.xlim(x_min, x_max)
# y_min = -0.1  # 最小值
# y_max = 1  # 最大值
# plt.ylim(y_min, y_max)

def hengxian(y_line, color, plt):
    cross_points = [cat for cat, val in zip(x, values) if val <= y_line]
    # 添加y=0.5的水平线
    if cross_points:
        max_x = x.index(cross_points[-1])  # 找到最后一个交叉点的索引
        plt.hlines(y=y_line, xmin=-5000,  xmax=max_x + 0.4, color=color, linestyle='--')
def shuxian(y_line, color, plt, x_bias=500, y_bias = -0.03):
    cross_points = [cat for cat, val in zip(x, values) if val <= y_line]
    # 添加y=0.5的水平线
    if cross_points:
        max_x = x.index(cross_points[-1])  # 找到最后一个交叉点的索引
        plt.vlines(x=max_x, ymin=-0.1, ymax=y_line, color=color, linestyle='--')
        # 在交点处添加x轴标记
        plt.text(max_x+x_bias, y_line+y_bias, max_x, fontsize=10)
        
# hengxian(0.5, 'red', plt)
# hengxian(0.8, 'orange', plt)
# hengxian(0.85, 'green', plt)    
# shuxian(0.5, 'red', plt)
# shuxian(0.8, 'orange', plt)
# shuxian(0.85, 'green', plt, x_bias=-4000, y_bias=0.02)

# 创建柱状图
plt.plot(x, values)
# 设置y轴刻度范围和间隔

# 添加标签和标题
plt.xlabel('')
plt.ylabel('Golden Score')

# 保存图像为文件
plt.savefig('bar_chart.png')

# 显示图像（可选）
plt.show()
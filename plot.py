import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 构建数据
data = {
    "Exposure-Outcome": ["BMI -> T2D"]*3 + ["LDL -> CAD"]*3 + ["Edu -> Smoke"]*3 + ["Height -> AD"]*3,
    "Method": ["IVW", "Weighted Median", "MR-Egger"]*4,
    "TwoSampleMR": [0.50, 0.48, 0.46, 0.40, 0.42, 0.38, -0.31, -0.29, -0.33, 0.02, 0.03, 0.05],
    "Automation": [0.49, 0.47, 0.45, 0.41, 0.43, 0.39, -0.30, -0.29, -0.32, 0.01, 0.02, 0.04]
}

df = pd.DataFrame(data)

# 设置风格
sns.set(style="whitegrid")

# X 轴位置
x_labels = df['Exposure-Outcome'].unique()
x = np.arange(len(x_labels))
width = 0.2  # 点偏移量

plt.figure(figsize=(10,6))

methods = df['Method'].unique()
colors = sns.color_palette("Set2", len(methods))

# 绘制点线
for i, method in enumerate(methods):
    subset = df[df['Method'] == method]
    # TwoSampleMR
    plt.scatter(x - width/2 + i*0.02, subset['TwoSampleMR'], color=colors[i], marker='o', label=f'TwoSampleMR - {method}')
    plt.plot(x - width/2 + i*0.02, subset['TwoSampleMR'], color=colors[i], linestyle='-')
    # Automation
    plt.scatter(x + width/2 + i*0.02, subset['Automation'], color=colors[i], marker='s', facecolors='none', label=f'Automation - {method}')
    plt.plot(x + width/2 + i*0.02, subset['Automation'], color=colors[i], linestyle='--')

# 美化图表
plt.xticks(x, x_labels, rotation=15)
plt.ylabel("Effect Size (β)", fontsize=12)
plt.xlabel("Exposure-Outcome Pair", fontsize=12)
plt.title("Method Consistency Comparison (Grouped Dot-Line Plot)", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

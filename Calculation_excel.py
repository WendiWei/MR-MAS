import pandas as pd
import numpy as np

# 原始 Excel 文件
file_path = "D:\研二上\paper\exposure_outcome_结果.xlsx"

# 读取
df = pd.read_excel(file_path)

# 计算 OR 及其置信区间
df["OR"] = np.exp(df["b"])
df["OR_lci95"] = np.exp(df["b"] - 1.96 * df["se"])
df["OR_uci95"] = np.exp(df["b"] + 1.96 * df["se"])

# 写回到同一个文件（覆盖保存）
df.to_excel(file_path, index=False)

print(f"✅ 已在 {file_path} 中追加 OR、OR_lci95、OR_uci95 列")

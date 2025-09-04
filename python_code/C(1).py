import pandas as pd

# 读取 Excel 文件
df = pd.read_excel("python_code//附件.xlsx")

# 统计每个编号出现的次数
counts = df["孕妇代码"].value_counts()

# 筛选出出现次数大于 3 次的编号
valid_ids = counts[counts > 3].index

# 从原始 DataFrame 中剔除出现次数不高于 3 次的编号对应的数据
filtered_df = df[df["孕妇代码"].isin(valid_ids)]

# 将结果保存为新的 Excel 文件
filtered_df.to_excel("文件//附件_剔除后数据.xlsx", index=False)

import pandas as pd

# 读取文件
file_path = '金工笔试第四题.xlsx'
data = pd.read_excel(file_path)
data.head() #数据预览

# 计算收盘价的平均数、方差和标准差
closing_price_mean = data['收盘价(元)'].mean()
closing_price_variance = data['收盘价(元)'].var()
closing_price_std = data['收盘价(元)'].std()

print(closing_price_mean)

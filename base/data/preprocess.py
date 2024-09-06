import pandas as pd

# Step 1: Load the Excel files
file_path1 = 'base/data/附件1.xlsx'
file_path2 = 'base/data/附件2.xlsx'

# Step 2: 读取所有表格数据
data_1 = pd.read_excel(file_path1, sheet_name=None)
data_2 = pd.read_excel(file_path2, sheet_name=None)

# Step 3: 处理销售价格区间，取中值
def process_price_range(price):
    if isinstance(price, str) and '-' in price:
        low, high = map(float, price.split('-'))
        return (low + high) / 2
    return price

# 应用区间处理函数
stats_2023_data_cleaned = data_2['2023年统计的相关数据'].copy()
stats_2023_data_cleaned['销售单价/(元/斤)'] = stats_2023_data_cleaned['销售单价/(元/斤)'].apply(process_price_range)

# Step 4: 合并2023年的种植数据与地块信息
merged_data = pd.merge(data_2['2023年的农作物种植情况'], data_1['乡村的现有耕地'], left_on='种植地块', right_on='地块名称')

# Step 5: 将作物的详情与统计数据进行合并
final_merged_data = pd.merge(merged_data, data_1['乡村种植的农作物'], left_on='作物编号', right_on='作物编号')
final_data = pd.merge(final_merged_data, stats_2023_data_cleaned, left_on='作物编号', right_on='作物编号')

# Step 6: 删除重复列，清理数据
final_cleaned_data = final_data.drop(columns=[
    '作物名称_x', '作物类型_x', '种植季次_x', '地块类型_x', '说明 ', '作物类型_y', '说明', '种植耕地'])

# Step 7: 重命名列名
final_cleaned_data.rename(columns={
    '作物名称_y': '作物名称',
    '地块类型_y': '地块类型',
    '种植季次_y': '种植季次',
    '地块名称': '耕地名称'
}, inplace=True)

# 保存结果为Excel文件
final_cleaned_data.to_excel('base/data/preprocessed.xlsx', index=False)

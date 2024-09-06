import pandas as pd
from openpyxl import load_workbook


def read_merged_cells(file_path, sheet_name):
    wb = load_workbook(file_path)
    sheet = wb[sheet_name]
    data = sheet.values
    columns = next(data)[0:]  # Get the column names
    df = pd.DataFrame(data, columns=columns)

    # Fill merged cell values
    for col in df.columns:
        df[col] = df[col].ffill()

    return df


# Load the Excel files
file_path_planting = "data/planting_statics-2023.xlsx"
file_path_land = "data/land_statics.xlsx"
file_path_stats = "data/general_statics-2023.xlsx"

# Read the data with merged cells handling
data_planting_2023 = read_merged_cells(file_path_planting, "2023年的农作物种植情况")
data_land = read_merged_cells(file_path_land, "乡村的现有耕地")
data_stats_2023 = read_merged_cells(file_path_stats, "2023年统计的相关数据")


# Process the sales price range and take the median
def process_price_range(price):
    if isinstance(price, str) and "-" in price:
        low, high = map(float, price.split("-"))
        return (low + high) / 2
    return price


# Apply the range processing function
data_stats_2023["销售单价/(元/斤)"] = data_stats_2023["销售单价/(元/斤)"].apply(
    process_price_range
)

stats_2023 = data_stats_2023[
    ["作物编号", "地块类型", "亩产量/斤", "种植成本/(元/亩)", "销售单价/(元/斤)"]
].drop_duplicates()

# Merging based on planting data in 2023
merged_data = data_planting_2023.copy()

# Merge in land data
merged_data = pd.merge(
    merged_data,
    data_land,
    how="left",
    left_on="地块编号",
    right_on="地块编号",
)

# Merge in price data
merged_data = pd.merge(
    merged_data,
    stats_2023,
    how="left",
    left_on=["作物编号", "地块类型"],
    right_on=["作物编号", "地块类型"],
)


# Remove useless columns
merged_data = merged_data.drop(
    columns=[
        "说明",
    ]
)

# Save the result as an Excel file
merged_data.to_excel("data/preprocess/pre-processed.xlsx", index=False)

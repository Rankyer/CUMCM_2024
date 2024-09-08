import pandas as pd
from openpyxl import load_workbook


def read_merged_cells(file_path: str, sheet_name: str) -> pd.DataFrame:
    wb = load_workbook(file_path)
    sheet = wb[sheet_name]
    data = sheet.values
    columns = next(data)
    df = pd.DataFrame(data, columns=columns)
    return df.ffill()


def strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    string_columns = df.select_dtypes(include=["object"])
    for col in string_columns.columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df


# Load the Excel files
file_paths = {
    "planting": "data/planting_stats-2023.xlsx",
    "land": "data/land_stats.xlsx",
    "stats": "data/general_stats-2023.xlsx",
}

# Read the data with merged cells handling
data_planting_2023 = strip_string_columns(
    read_merged_cells(file_paths["planting"], "2023年的农作物种植情况")
)
data_land = strip_string_columns(
    read_merged_cells(file_paths["land"], "乡村的现有耕地")
)
data_stats_2023 = strip_string_columns(
    read_merged_cells(file_paths["stats"], "2023年统计的相关数据")
)


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

# Labeling Fields with Field ID
data_land["地块编号"] = data_land.index + 1

# Merge in land data
merged_data = pd.merge(
    merged_data,
    data_land,
    how="left",
    left_on="地块名称",
    right_on="地块名称",
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

# Define a mapping for column names
column_mapping = {
    "地块名称": "Field Name",
    "地块编号": "Field ID",
    "作物编号": "Crop ID",
    "作物名称": "Crop Name",
    "作物类型": "Crop Type",
    "种植面积/亩": "Planting Area",
    "种植季次": "Season",
    "地块类型": "Field Type",
    "地块面积/亩": "Field Area",
    "亩产量/斤": "Per Yield",
    "种植成本/(元/亩)": "Per Cost",
    "销售单价/(元/斤)": "Per Price",
}

# Rename columns in the DataFrames
merged_data.rename(columns=column_mapping, inplace=True)

# Replace value in the column
merged_data["Season"] = merged_data["Season"].replace({"单季": "第一季"})

# Add a new column for the Season ID
season_mapping = {"第一季": 1, "第二季": 2}
merged_data["Season ID"] = merged_data["Season"].map(season_mapping)

# Calculate the yield, total cost, total revenue and profit
merged_data["Yield"] = merged_data["Planting Area"] * merged_data["Per Yield"]
merged_data["Cost"] = merged_data["Planting Area"] * merged_data["Per Cost"]

yield_selling_ratio = 0.8
merged_data["Selling"] = merged_data["Yield"] * yield_selling_ratio
merged_data["Revenue"] = merged_data["Selling"] * merged_data["Per Price"]
merged_data["Profit"] = merged_data["Revenue"] - merged_data["Cost"]

# Reorder the columns
column_order = [
    "Field ID",
    "Field Name",
    "Crop ID",
    "Crop Name",
    "Crop Type",
    "Planting Area",
    "Season ID",
    "Season",
    "Field Type",
    "Field Area",
    "Per Yield",
    "Per Cost",
    "Per Price",
    "Yield",
    "Cost",
    "Selling",
    "Revenue",
    "Profit",
]

merged_data = merged_data[column_order]

# Save the result as an Excel file
merged_data.to_csv("data/preprocess/pre-processed.csv", index=False)

temp = merged_data.copy()

# 统计每种作物的种植面积和售价与种植成本
crop_stats = temp.groupby("Crop ID").agg(
    {
        "Planting Area": "sum",
        "Per Price": "mean",
        "Per Cost": "mean",
        "Yield": "sum",
        "Cost": "sum",
        "Selling": "sum",
        "Revenue": "sum",
        "Profit": "sum",
    }
)
crop_stats.reset_index(inplace=True)

# 四舍五入所有数据
crop_stats = crop_stats.round(2)
crop_stats.drop(columns=["Selling", "Revenue", "Profit"], inplace=True)
crop_stats.to_csv("data/preprocess/temp.csv", index=False)

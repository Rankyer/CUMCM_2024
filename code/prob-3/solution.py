# %%
import pandas as pd
import numpy as np
import datetime
from planning import planning

stats_2023 = pd.read_csv("../../data/preprocess/pre-processed.csv")
stats_2023.dtypes

# %%
fields_id = list(map(int, stats_2023["Field ID"].unique()))  # 所有的地块
crops_id = list(map(int, stats_2023["Crop ID"].unique()))  # 所有的作物
seasons_id = list(map(int, stats_2023["Season ID"].unique()))  # 所有的时节
fields_id.sort()
crops_id.sort()
seasons_id.sort()
years = range(2023, 2031)  # 要优化的年份


def pop_ndarray(arr, element_to_pop):
    index = np.where(arr == element_to_pop)[0]
    if index.size > 0:
        arr = np.delete(arr, index)
    return arr


# A类粮食作物的集合
grains_A = stats_2023[
    (stats_2023["Crop Type"] == "粮食") | (stats_2023["Crop Type"] == "粮食（豆类）")
]["Crop ID"].unique()
grains_A = pop_ndarray(grains_A, 16)
grains_A.sort()

# B类粮食作物的集合
grains_B = np.array([16])

# A类蔬菜作物的集合
vege_A = stats_2023[
    (stats_2023["Crop Type"] == "蔬菜") | (stats_2023["Crop Type"] == "蔬菜（豆类）")
]["Crop ID"].unique()
for i in range(3):
    vege_A = pop_ndarray(vege_A, 35 + i)
vege_A.sort()

# B类蔬菜作物的集合
vege_B = np.array([35, 36, 37])

# 食用菌作物的集合
mush = stats_2023[stats_2023["Crop Type"] == "食用菌"]["Crop ID"].unique()
mush.sort()

# 豆类作物的集合
beans = stats_2023[
    (stats_2023["Crop Type"] == "粮食（豆类）")
    | (stats_2023["Crop Type"] == "蔬菜（豆类）")
]["Crop ID"].unique()
beans.sort()

# 第 i 个地块的类型（如平旱地、梯田、山坡地、智能大棚、普通大棚、水浇地）
t_i = {
    i: stats_2023[stats_2023["Field ID"] == i]["Field Type"].values[0]
    for i in fields_id
}

# 第 i 个地块在第 s 季可种植的作物集合
T_hat_i_s: dict[tuple : np.ndarray] = {}


for i in fields_id:
    if t_i[i] in ["平旱地", "梯田", "山坡地"]:
        T_hat_i_s[i, 1] = grains_A
        T_hat_i_s[i, 2] = np.array([])
    elif t_i[i] in ["水浇地"]:
        T_hat_i_s[i, 1] = np.concatenate([grains_B, vege_A])
        T_hat_i_s[i, 2] = vege_B
    elif t_i[i] in ["普通大棚"]:
        T_hat_i_s[i, 1] = vege_A
        T_hat_i_s[i, 2] = mush
    elif t_i[i] in ["智慧大棚"]:
        T_hat_i_s[i, 1] = vege_A
        T_hat_i_s[i, 2] = vege_A

# 市场互替性系数
R_ij = {}  # 作物 i 对作物 j 在市场上的互替性系数

# 将所有作物对初始化为 0
for i in crops_id:
    for j in crops_id:
        if i != j:
            R_ij[i, j] = 0


# 对于同类型的作物，其市场价格接近，互替性系数越高
# 定义一个函数来计算互替性系数
def calculate_substitutability(price_i, price_j):
    # 计算价格差距的百分比
    price_diff_percentage = abs(price_i - price_j) / ((price_i + price_j) / 2)
    # 使用一个简单的公式来量化互替性系数
    # 这里假设价格差距越小，互替性越高
    substitutability = max(0, 1 - price_diff_percentage)
    return substitutability


# 计算同类型作物之间的互替性系数
def calculate_R_ij_for_type(crop_ids):
    for i in crop_ids:
        for j in crop_ids:
            if i != j:
                price_i = stats_2023[stats_2023["Crop ID"] == i]["Per Price"].values[0]
                price_j = stats_2023[stats_2023["Crop ID"] == j]["Per Price"].values[0]
                if price_i is not None and price_j is not None:
                    R_ij[i, j] = calculate_substitutability(price_i, price_j)


# 假设 grains_A 是同类型作物的集合
calculate_R_ij_for_type(grains_A)
calculate_R_ij_for_type(grains_B)
calculate_R_ij_for_type(vege_A)
calculate_R_ij_for_type(vege_B)
calculate_R_ij_for_type(mush)
calculate_R_ij_for_type(beans)

print(R_ij)

# %%
dfs = []
times = 30  # 重复次数


def expected_selling(percost, price):
    return 8122.041 + 1.385 * percost - 1.2894625 * price


for i in range(times):
    # 使用当前时间作为随机数种子
    np.random.seed(datetime.datetime.now().second)

    S = {}  # 期望产量
    Y = {}  # 单位面积产量
    P = {}  # 售价
    C = {}  # 单位面积成本

    for j in crops_id:
        Y[j, 2023] = stats_2023[stats_2023["Crop ID"] == j]["Per Yield"].values[0]
        P[j, 2023] = stats_2023[stats_2023["Crop ID"] == j]["Per Price"].values[0]
        C[j, 2023] = stats_2023[stats_2023["Crop ID"] == j]["Per Cost"].values[0]
        S[j, 2023] = expected_selling(C[j, 2023], P[j, 2023])

    for j in crops_id:
        # 小麦和玉米的预期销售量有平均5%-10%的年增长
        for k in years:

            # 农作物亩产量可能有±10%的变化.
            increment = np.random.uniform(-0.1, 0.1)
            Y[j, k + 1] = Y[j, k] * np.random.normal(1 + increment, 0.01)

            # 种植成本每年增长约5%。
            increment = 0.05
            C[j, k + 1] = C[j, k] * np.random.normal(1 + increment, 0.01)

            # 蔬菜类作物价格每年增长约5%。
            if j in vege_A or j in vege_B:
                increment = 0.05
                P[j, k + 1] = P[j, k] * np.random.normal(1 + increment, 0.01)
            else:
                P[j, k + 1] = P[j, k]

            # 食用菌价格每年下降1%-5%。
            if j in mush:
                increment = np.random.uniform(-0.05, -0.01)
                P[j, k + 1] = P[j, k] * np.random.normal(1 + increment, 0.01)
            else:
                P[j, k + 1] = P[j, k]

            # 羊肚菌的销售价格每年下降幅5%
            if j == 41:
                increment = -0.05
                P[j, k + 1] = P[j, k] * (1 + increment)
            else:
                P[j, k + 1] = P[j, k]

            S[j, k + 1] = expected_selling(C[j, k + 1], P[j, k + 1])

            if j in [6, 7]:
                increment = np.random.uniform(0.05, 0.1)
                S[j, k + 1] = S[j, k] * np.random.normal(1 + increment, 0.01)

            # 其他作物的预期销售量变化约±5%。
            else:
                increment = np.random.uniform(-0.05, 0.05)
                S[j, k + 1] = S[j, k] * np.random.normal(1 + increment, 0.01)

    dfs.append(
        planning(
            fields_id,
            crops_id,
            seasons_id,
            years,
            t_i,
            T_hat_i_s,
            R_ij,
            grains_A,
            grains_B,
            vege_A,
            vege_B,
            mush,
            beans,
            stats_2023,
            [
                S,
                Y,
                P,
                C,
            ],
        )
    )

# %%
# 修正后的代码
df = pd.concat(dfs)

# 计算平均值时需要指定哪些列进行聚合
df = df.groupby(["Field", "Crop", "Season", "Year"]).agg("mean").reset_index()
df = df[~((df["Season"] == 2) & (df["Field"] < 27))]  # 去掉单季地块第二季度的数据

# 获取不包括 2023 年的唯一年份
years = df["Year"].unique()
years = [year for year in years if year != 2023]


# 创建一个 Excel writer 对象
with pd.ExcelWriter(f"exp-{times}.xlsx") as writer:
    for year in years:
        # 过滤特定年份的数据
        df_year = df[df["Year"] == year]

        # 透视表，Field 作为行，Crop 作为列
        pivot_table = df_year.pivot_table(
            index=["Season", "Field"], columns="Crop", values="Planting", aggfunc="sum"
        ).round(
            1
        )  # 保留一位小数

        # 将透视表写入 Excel 的 sheet
        pivot_table.to_excel(writer, sheet_name=str(year))

print(f"Exported to exp-{times}.xlsx")



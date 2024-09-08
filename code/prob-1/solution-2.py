# %%
import pandas as pd

stats_2023 = pd.read_csv("../../data/preprocess/pre-processed.csv")
stats_2023.dtypes

# %% [markdown]
# # 初始化

# %%
import pulp
import numpy as np

# %% [markdown]
# # 定义辅助变量

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

S: dict[tuple : np.int64] = {}  # 期望产量
Y: dict[tuple : np.int64] = {}  # 单位面积产量
P: dict[tuple : np.int64] = {}  # 售价
C: dict[tuple : np.int64] = {}  # 单位面积成本

for j in crops_id:
    # 所有参数与 2023 年的保持一致
    for k in years:
        S[j, k] = stats_2023.groupby(["Crop ID"])["Selling"].agg("sum").values[j - 1]
        Y[j, k] = stats_2023[stats_2023["Crop ID"] == j]["Per Yield"].values[0]
        P[j, k] = stats_2023[stats_2023["Crop ID"] == j]["Per Price"].values[0]
        C[j, k] = stats_2023[stats_2023["Crop ID"] == j]["Per Cost"].values[0]

# %% [markdown]
# # 定义决策变量

# %%
# A^1_{ijks}：第 k 年第 s 季度的第 i 块地，第 j 作物的种植面积，若为不为大棚地块则为 0
# A^2_{ijks}：第 k 年第 s 季度的第 i 块地，第 j 作物的种植面积，若为大棚地块则为 0
# A_{ijks}：第 k 年第 s 季度的第 i 块地，第 j 作物的种植面积
A_ijks = pulp.LpVariable.dicts(
    "A",
    [
        (i, j, k, s)
        for i in fields_id
        for j in crops_id
        for k in years
        for s in seasons_id
    ],
    lowBound=0,
    cat=pulp.LpContinuous,
)


# Rename
for i in fields_id:
    for j in crops_id:
        for k in years:
            for s in seasons_id:
                A_ijks[i, j, k, s].name = f"A_{i}_{j}_{k}_{s}"


# A_{ijk} = \sum_s A^2_{ijks} + A^1_{ij(k-1)2} + A^1_{ijks}
A_ijk: dict[tuple : pulp.LpVariable] = {}
for i in fields_id:
    for j in crops_id:
        for k in years:
            if k == 2023:
                continue
            A_ijk[i, j, k] = pulp.lpSum([A_ijks[i, j, k, s] for s in seasons_id])


# 2023 年的 A_{ijks} 已知
for i in fields_id:
    for j in crops_id:
        for s in seasons_id:
            matching_rows = stats_2023[
                (stats_2023["Field ID"] == i)
                & (stats_2023["Crop ID"] == j)
                & (stats_2023["Season ID"] == s)
            ]

            area = (
                matching_rows["Planting Area"].values[0]
                if not matching_rows.empty
                else 0
            )

            A_ijks[i, j, 2023, s].lowBound = area
            A_ijks[i, j, 2023, s].upBound = area

# 2023 年的 A_{ijk} 已知
for i in fields_id:
    for j in crops_id:
        A_ijk[i, j, 2023] = pulp.lpSum([A_ijks[i, j, 2023, s] for s in seasons_id])

B_ijks = pulp.LpVariable.dicts(
    "B",
    [
        (i, j, k, s)
        for i in fields_id
        for j in crops_id
        for k in years
        for s in seasons_id
    ],
    lowBound=0,
    cat=pulp.LpBinary,
)

A_star = {}  # 地块面积

for i in fields_id:
    A_star[i] = stats_2023[stats_2023["Field ID"] == i]["Field Area"].values[0]

# %% [markdown]
# # 定义目标函数

# %%
# 创建线性规划问题
prob = pulp.LpProblem("Crop_Optimization", pulp.LpMaximize)

alpha = 0.5  # 情况二

# 定义辅助变量 Z_ijk
Z_ijk = pulp.LpVariable.dicts(
    "Z",
    [(i, j, k) for i in fields_id for j in crops_id for k in years],
    lowBound=0,
    cat=pulp.LpContinuous,
)

# 线性化的目标函数
L = pulp.lpSum(
    [
        (
            Y[j, k] * A_ijk[i, j, k] * P[j, k]
            - C[j, k] * A_ijk[i, j, k]
            - alpha * Z_ijk[i, j, k] * P[j, k]
        )
        for i in fields_id
        for j in crops_id
        for k in years
    ]
)

# 设置目标函数
prob += L

# %%
# 辅助约束： Z_ijk >= Y_jk * A_ijk - S_jk
for i in fields_id:
    for j in crops_id:
        for k in years:
            prob += Z_ijk[i, j, k] >= Y[j, k] * A_ijk[i, j, k] - S[j, k]

# 添加约束 Z_ijk >= 0 已经通过 LpVariable 的 lowBound=0 实现

# %%
# 辅助约束
M1 = 10000  # 一个很大的数

for i in fields_id:
    for j in crops_id:
        for k in years:
            for s in seasons_id:
                prob += A_ijks[i, j, k, s] <= B_ijks[i, j, k, s] * M1

# %%
# 约束：一个地块上每种作物合起来的种植面积不超过相应地块的总面积
for i in fields_id:
    for s in seasons_id:
        for k in years:
            prob += (
                pulp.lpSum([A_ijks[i, j, k, s] for j in T_hat_i_s[i, s]]) <= A_star[i]
            )
            for j in crops_id:
                if j not in T_hat_i_s[i, s]:
                    prob += A_ijks[i, j, k, s] == 0

# %%
# 约束：不能重茬种植
for i in fields_id:
    for j in crops_id:
        for k in years:
            if k >= years[-1]:  # 最后一年不需要考虑后面的年份
                continue

            # 如果作物 j 在第 i 块地的第 1 季度被种植
            if j in T_hat_i_s[i, 1] and j not in T_hat_i_s[i, 2]:
                prob += B_ijks[i, j, k, 1] + B_ijks[i, j, k + 1, 1] <= 1
            # 如果作物 j 在第 i 块地的第 1 和第 2 季度都被种植
            elif j in T_hat_i_s[i, 1] and j in T_hat_i_s[i, 2]:
                prob += B_ijks[i, j, k, 1] + B_ijks[i, j, k, 2] <= 1

# %%
# 约束：每个地块（含大棚）的所有土地三年内至少种植一次豆类作物
for i in fields_id:
    for k in years:
        if k >= years[-2]:  # 最后两年不需要考虑后面的年份
            continue

        prob += (
            pulp.lpSum(
                [
                    B_ijks[i, j, kk, s]
                    for kk in range(k, k + 3)
                    for s in seasons_id
                    for j in beans
                    if j in T_hat_i_s[i, s]
                ]
            )
            >= 1
        )

        for j in beans:
            for s in seasons_id:
                if j in T_hat_i_s[i, s]:
                    # 如果种植，则种植整个地块面积
                    prob += A_ijks[i, j, k, s] == A_star[i] * B_ijks[i, j, k, s]

# %%
# 约束：第一季种了水稻的水浇地，第二季不能种 B 类蔬菜
for i in fields_id:
    for k in years:
        if t_i[i] == "水浇地":
            for j1 in grains_B:
                for j2 in vege_B:
                    prob += B_ijks[i, j1, k, 1] + B_ijks[i, j2, k, 2] <= 1

# %%
# 约束：第一季种了 A 类蔬菜的水浇地，第二季的 B 类蔬菜不能混种在一个地块上
for i in fields_id:
    for k in years:
        if t_i[i] == "水浇地":
            for j2 in vege_B:
                prob += A_ijks[i, j2, k, 2] == A_star[i] * B_ijks[i, j2, k, 2]

# %% [markdown]
# # 模型求解

# %%
prob.solve()

# %% [markdown]
# # 整理结果

# %%
A_ijks_output = {}

for v in prob.variables():
    if v.name.startswith("A_star"):
        continue

    if v.name.startswith("A"):
        name = v.name.replace("A_", "")
        A_ijks_output[tuple(map(int, name.split("_")))] = v.varValue


A_ijks_output = dict(
    sorted(
        A_ijks_output.items(),
        key=lambda item: (
            int(item[0][2]),  # year
            int(item[0][3]),  # season
            int(item[0][0]),  # field
            int(item[0][1]),  # crop
        ),
    )
)

for key, value in A_ijks_output.items():
    print(key, value)

# %%
# 将字典转换为 DataFrame
df = pd.DataFrame(
    [
        (field, crop, year, season, value)
        for (field, crop, year, season), value in A_ijks_output.items()
    ],
    columns=["Field", "Crop", "Year", "Season", "Planting"],
)

# 计算平均值时需要指定哪些列进行聚合
df = df.groupby(["Field", "Crop", "Season", "Year"]).agg("mean").reset_index()
df = df[~((df["Season"] == 2) & (df["Field"] < 27))]  # 去掉单季地块第二季度的数据

# 获取不包括 2023 年的唯一年份
years = df["Year"].unique()
years = [year for year in years if year != 2023]


# 创建一个 Excel writer 对象
with pd.ExcelWriter(f"part-2.xlsx") as writer:
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

print(f"Exported to part-2.xlsx")



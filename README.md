
# CUMCM 2024 C

## 题干

这道题目旨在研究一个位于华北山区的乡村在2024至2030年间的农作物种植优化方案。具体内容可以总结为以下几个关键点：

### 背景信息
- **耕地资源**：乡村拥有1201亩耕地，分为平旱地、梯田、山坡地、智能大棚、普通大棚、水浇地。
- **种植地块类型和时节限制**：
    - 平旱地、梯田和山坡地每年适宜单季种植 A 类粮食作物
    - 水浇地每年可以单季种植 B 类粮食作物（只有水稻）或两季种植蔬菜作物
        - 规定 大白菜、白萝卜和红萝卜 为 B 类蔬菜，其他蔬菜为 A 类蔬菜
            - 若在某块水浇地种植两季蔬菜，第一季只能种植 A 类蔬菜；第二季只能种植 B 类蔬菜中的一种（不能混种）。
            - 普通大棚每年第一季只能种植 A 类蔬菜，第二季只能种植食用菌
            - 智慧大棚每年可种植两季 A 类蔬菜
- **多年连续种植限制**
    - 每个地块（含大棚）的所有土地三年内至少种植一次豆类作物
    - 每种作物在同一地块（含大棚）都不能连续重茬种植
- **种植空间关系限制**
    - 每种作物每季的种植地不能太分散
    - 每种作物在单个地块种植的面积不能太小
    - 同一地块每季**可以**合种不同的作物

- **一年多季耕地的时间说明**：
    - 大棚的第一季通常是在每年的5月至9月前后，第二季是在9月至下一年4月前后
    - 也就是说，大棚在今年第二季决定的种植作物和面积影响的是下一年的作物产量

### 研究问题
1. **最优种植方案（2024-2030年）**：
   - **情况1**：如果某种作物的总产量超过预期销售量，超出部分滞销。
   - **情况2**：超出部分按50%降价出售。

2. **考虑不确定性因素的最优种植方案**：
   - 小麦和玉米的销售量有5%-10%的年增长，其他作物的销售量变化约±5%。
   - 农作物亩产量可能有±10%的变化，种植成本每年增长约5%。
   - 蔬菜类作物价格每年增长约5%，食用菌价格每年下降1%-5%。

3. **综合考虑相关性因素的最优种植策略**：
   - 考虑农作物之间的可替代性和互补性，及其销售量、价格和成本之间的相关性。

---


## 模型

### 符号定义

- $S_{jk}$：第 $k$ 年第 $j$ 作物的期望销量
- $Y_{jk}$：第 $k$ 年第 $j$ 作物的单位面积产量
- $P_{jk}$：第 $k$ 年第 $j$ 作物的售价
- $C_{jk}$：第 $k$ 年第 $j$ 作物的单位面积成本
- $A_{i}^*$： 第 $i$ 个地块的面积
- $A^1_{ijks}$：第 $k$ 年第 $s$ 季度的第 $i$ 大棚上，第 $j$ 作物的种植面积
- $A^2_{ijks}$：第 $k$ 年第 $s$ 季度的第 $i$ 非大棚地块上，第 $j$ 作物的种植面积
- $A_{ijk}$：第 $k$ 年第 $i$ 地块上，第 $j$ 作物的总种植面积，定义为：
  $$
  A_{ijk} = \sum_s A^2_{ijks} + A^1_{ij(k-1)2} + A^1_{ijks}
  $$
- $t_i$： 第$i$个地块的类型（如平旱地、梯田、山坡地、智能大棚、普通大棚、水浇地）
- $\hat{T}_{is}$：第 $i$ 个地块在第 $s$ 季可种植的作物集合
- $\text{Beans}$：豆类作物的集合
- $\text{Grains}_A$：A类粮食作物的集合，即除了水稻之外的粮食作物
- $\text{Grains}_B$：B类粮食作物的集合，即水稻
- $\text{Vege}_A$：A类蔬菜的集合，即除了大白菜、白萝卜和红萝卜之外的蔬菜
- $\text{Vege}_B$：B类蔬菜的集合，即大白菜、白萝卜和红萝卜
- $\text{Mush}$：食用菌的集合

### 决策变量

- $A^1_{ijks}$：第 $k$ 年第 $s$ 季度的第 $i$ 大棚上，第 $j$ 作物的种植面积
- $A^2_{ijks}$：第 $k$ 年第 $s$ 季度的第 $i$ 非大棚地块上，第 $j$ 作物的种植面积

### 目标函数

#### 情况1：如果某种作物的总产量超过预期销售量，超出部分滞销

$$
L_1 = \sum_{ijk} \left( \min \left( S_{jk}, Y_{jk} \cdot A_{ijk} \right) \cdot P_{jk} - C_{jk} \cdot A_{ijk} \right)
$$

#### 情况2：超出部分按50%降价出售

$$
L_2 = \sum_{ijk} \left( Y_{jk} \cdot A_{ijk} \cdot P_{jk} - C_{jk} \cdot A_{ijk} - 0.5 \cdot \max \left( 0, Y_{jk} \cdot A_{ijk} - S_{jk} \right) \cdot P_{jk} \right)
$$

### 约束条件
- 每种作物合起来的种植面积不超过相应地块的总面积
    $$
    \sum_j A_{ijks} \leq A_i^* \quad \forall i,k \text{ and } j \in \hat{T}_{is}
    $$

- 每种作物在同一地块（含大棚）都不能连续重茬种植
    $$
    A_{ij(k-1)s}+A_{ijks} \leq min(A_{ij(k-1)s},A_{ijks}) \quad \forall i,k \text{ and } j \in \hat{T}_{is}
    $$

- 每个地块（含大棚）的所有土地三年内至少种植一次豆类作物
    $$
    max(A_{ij(k-2)s}, A_{ij(k-1)s},A_{ijks}) = A_i^* \quad \forall i,k \text{ and } j \in \hat{T}_{is}
    $$

- 每种作物在单个地块（含大棚）种植的面积不宜太小
    $$
    A_{ijks}^{n} \geq M \times A_i^*  \quad \text{if } A_{ijks}^{n} \neq 0 \qquad \forall i,k \text{ and } j \in \hat{T}_{is},n \in \{1,2\}
    $$

- 种植的地块类型限制和季节限制
    $$
    \hat{T}_i =
    \begin{cases}
    \begin{aligned}
        & \hat{T}_{i1} = \text{Grains}_A, \quad \hat{T}_{i2} = \phi, \quad & \text{if } t_i \in \{\text{平旱地}, \text{梯田}, \text{山坡地}\} \\
    \end{aligned} \\
    \begin{aligned}
        & \hat{T}_{i1} = \text{Grains}_B \quad \text{或} \quad \hat{T}_{i1} = \text{Vege}_A, \quad \hat{T}_{i2} = \text{Vege}_B, \quad & \text{if } t_i \in \{\text{水浇地}\} \\
    \end{aligned} \\
    \begin{aligned}
        & \hat{T}_{i1} = \text{Vege}_A, \quad \hat{T}_{i2} = \text{Mush}, \quad & \text{if } t_i \in \{\text{普通大棚}\} \\
    \end{aligned} \\
    \begin{aligned}
        & \hat{T}_{i1} = \hat{T}_{i2} = \text{Vege}_A, \quad & \text{if } t_i \in \{\text{智慧大棚}\} \\
    \end{aligned} \\
    \end{cases}
    $$

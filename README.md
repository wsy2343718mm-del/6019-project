# Inverse Reinforcement Learning on GridWorld

本项目实现了四种经典逆向强化学习(IRL)算法，在10×10 GridWorld导航环境中进行完整的实验评估与对比分析。

## 项目概述

**课程**: COMP 6019 - 强化学习  
**题目**: Project 3 - Inverse Reinforcement Learning and Preference-Based Learning  
**难度**: ★★★★☆ (4.1/5)

本项目对应 Lecture 7 (Preference-based Learning) 和 Lecture 8 (Inverse RL)，探索奖励模糊性等深层理论问题，并连接现代 RLHF 技术。

## 环境配置

### 环境规格

| 参数 | 值 |
|------|-----|
| 网格大小 | 10×10 (100状态) |
| 目标位置 | (9, 9) |
| 障碍物数量 | 22个 |
| 随机性 | 10% 动作噪声 |
| 折扣因子 | γ = 0.9 |
| 最大步数 | 200 |

### 特征表示

- **RBF特征** (3维): 目标接近度 + 障碍物距离 + 常数偏置
- **坐标特征** (2维): 归一化行列坐标
- **One-hot特征** (100维): 状态独热编码

## 算法实现

| 算法 | 方法 | 求解器 |
|------|------|--------|
| **LP-IRL** | 线性规划特征匹配 | SciPy HiGHS |
| **MM-IRL** | 最大边距优化 | CVXPY + SCS (SOCP) |
| **MaxEnt-IRL** | 最大熵软值迭代 | NumPy (梯度优化) |
| **Preference-BT** | Bradley-Terry偏好学习 | NumPy (梯度优化) |

## 项目结构

```
6019_GridWorld_QWEN/
├── env/
│   └── gridworld.py           # GridWorld环境 (转移矩阵P, 特征Phi)
├── data/
│   └── generate_expert.py     # 专家轨迹与偏好数据生成
├── irl/
│   ├── lp_irl.py              # 线性规划IRL
│   ├── mm_irl.py              # 最大边距IRL
│   ├── maxent_irl.py          # 最大熵IRL
│   ├── preference_irl.py      # Bradley-Terry偏好学习
│   └── utils.py               # 特征期望计算工具
├── policy/
│   └ train_policy.py          # 值迭代策略训练器
├── utils/
│   └ viz.py                   # 可视化工具
├── tests/                     # 测试与诊断脚本
├── exp/
│   └ ablation_noise.py        # 噪声消融实验
├── figures/                   # 实验图表输出
├── main_lp_mm_irl.py          # LP/MM-IRL执行脚本
├── main_maxent_pref_irl.py    # MaxEnt/Pref执行脚本
├── main_comparison.py         # 综合对比实验
├── requirements.txt
└── README.md
```

## 安装与运行

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行步骤

**Step 1: 生成专家数据**
```bash
python data/generate_expert.py
```
生成 `data/expert_dataset.pkl`，包含:
- 500条干净专家轨迹
- 500条噪声轨迹 (15%动作翻转)
- 300个Bradley-Terry偏好对

**Step 2: 运行IRL算法**
```bash
# LP-IRL 和 MM-IRL
python main_lp_mm_irl.py

# MaxEnt-IRL 和 Preference-BT
python main_maxent_pref_irl.py
```

**Step 3: 综合评估与对比**
```bash
python main_comparison.py
```
生成:
- 策略成功率对比图
- 噪声消融曲线
- 奖励场热力图
- 对比数据保存至 `data/comparison_results.pkl`

```

## 实验结果

### 主实验结果 (固定起点 (0,0), 100次评估)

| 方法 | 成功率 | 平均步数 | Pearson相关 | 目标排名 |
|------|--------|----------|-------------|----------|
| Ground Truth | 100% | 21.0 | 1.000 | 1/100 |
| LP-IRL | 100% | 20.8 | 0.502 | 1/100 |
| MM-IRL | 100% | 20.8 | 0.074 | 1/100 |
| MaxEnt-IRL | 100% | 20.6 | 0.368 | 1/100 |
| Preference-BT | 100% | 20.6 | 0.293 | 1/100 |
| Random | 0% | - | - | - |

### 噪声鲁棒性消融

| 噪声水平 | LP-IRL | MM-IRL | MaxEnt | Preference-BT |
|----------|--------|--------|--------|---------------|
| 0.0 | 99% | 100% | 100% | 100% |
| 0.1 | 0% | 100% | 100% | 100% |
| 0.2-0.4 | 0-1% | 100% | 100% | 100% |

## 关键发现

1. **所有IRL算法实现100%策略成功率**，验证了从演示中恢复奖励的可行性

2. **奖励模糊性现象验证**: MM-IRL Pearson相关仅0.074但成功率100%，证明"不同奖励可产生相同策略"

3. **噪声鲁棒性差异显著**: LP-IRL对噪声最敏感，MM-IRL/MaxEnt/Pref-BT在所有噪声水平下保持100%

4. **与RLHF的连接**: Preference-BT直接连接现代基于人类反馈的强化学习

## 团队分工

| 学生 | 职责 |
|------|------|
| Student 1 | 环境 + 专家轨迹/偏好数据生成 |
| Student 2 | LP-IRL + MM-IRL 实现 |
| Student 3 | MaxEnt-IRL + Preference-BT 实现 |
| Student 4 | 策略训练 + 对比实验 + 报告 |

## License

本项目为课程作业，仅供学习参考。
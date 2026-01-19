# 校园配送系统离散事件仿真

## 项目概述

本项目实现了一个**高保真、面向对象的离散事件仿真(DES)系统**，用于解决校园异构配送系统的动态车辆路径问题(HFVRPTW)。

### 基于论文规格
- **仿真时间**: 11:30-12:30 (午高峰1小时)
- **Rolling Horizon Control (RHC)**: 每5分钟触发一次调度
- **异构车队**: 人类骑手(20辆)、AGV(10辆)、无人机(6架)

## 项目结构

```
project/
├── config.py           # 全局参数配置
├── entities.py         # 实体类: Order, Agent, Rider, Drone, AGV, Bundle
├── generator.py        # 需求生成 (NHPP非齐次泊松过程)
├── solver.py           # 调度算法 (两阶段启发式)
├── simulator.py        # 主仿真循环和环境类
├── analysis.py         # 可视化和KPI计算
├── main.py             # 程序入口
├── output/             # 输出目录（图表）
│   ├── simulation_report.png
│   ├── agent_distribution.png
│   ├── delivery_time_hist.png
│   └── order_cumulative.png
└── README.md           # 本文件
```

## 核心算法

### 1. 需求生成 (NHPP)
- 基于分段常数强度函数的非齐次泊松过程
- 每10分钟区间订单量: 55→75→98→107(峰值)→84→64
- 餐厅分布: R1(30%), R2(25%), R3(25%), R4(20%)
- 宿舍分布: 各25%均匀分布

### 2. 两阶段启发式调度
**阶段1: 批次化 (Clustering)**
- 按餐厅分组订单
- 按截止时间接近度(5分钟内)划分子批次
- 批次大小 ≤ 5 (人类骑手容量)

**阶段2: 贪婪分配 (Greedy Utility)**
- 紧急订单(剩余<15min) & 单个: Drone > AGV > Human
- 大批量(≥3): 仅Human Rider
- 小批量/单个: AGV > Human

### 3. 约束条件
- **硬性截止时间**: 30分钟
- **无人机禁飞区**: R2 ↔ D3 路径禁止飞行
- **往返逻辑**: 代理锁定时间 = Hub→餐厅→宿舍→Hub

## 运行方式

### 基本运行
```bash
python main.py
```

### 命令行参数
```bash
python main.py --verbose          # 显示详细日志
python main.py --save-plots       # 保存图表
python main.py --output-dir ./out # 指定输出目录
python main.py --no-show          # 不显示图表(无GUI环境)
python main.py --seed 123         # 设置随机种子
```

### 完整示例
```bash
python main.py --save-plots --output-dir ./output --no-show
```

## 仿真结果 (示例)

| 指标 | 值 |
|------|-----|
| 总订单数 | 483 |
| 完成订单 | 250 |
| 准时率 | 51.76% |
| 平均成本/单 | 0.678 |
| 平均延迟 | 0.00 min |

## 可视化输出

1. **simulation_report.png**: 综合报告(4图合一)
   - 需求热力图
   - 车队使用堆叠面积图
   - 订单状态饼图
   - KPI仪表盘

2. **agent_distribution.png**: 各代理类型订单分布

3. **delivery_time_hist.png**: 配送时间分布直方图

4. **order_cumulative.png**: 订单状态变化曲线

## 扩展功能

### 敏感性分析
```python
from main import run_sensitivity_analysis
run_sensitivity_analysis()  # 测试不同RHC间隔
```

### 蒙特卡洛仿真
```python
from main import run_monte_carlo_simulation
run_monte_carlo_simulation(n_runs=10)  # 多次运行统计分析
```

## 数学模型参考

详见论文中的以下章节:
- 3.1 Problem Decomposition
- 3.2 NHPP Demand Generation
- 3.3 Two-Stage Heuristic Dispatch
- 3.4 Rolling Horizon Control Framework

## 依赖项

```
numpy>=1.20.0
matplotlib>=3.5.0
```

安装:
```bash
pip install numpy matplotlib
```

## 作者

数学建模竞赛 - Problem B: Campus Delivery System Design

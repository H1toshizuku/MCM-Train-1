"""
config.py - 全局参数与常量配置

This module contains all global parameters and constants for the 
Heterogeneous Campus Delivery System simulation.

Mathematical Modeling Contest - Problem B: Campus Delivery System Design
"""

import numpy as np

# ============================================================================
# Reproducibility Seed
# ============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# Simulation Time Parameters (单位: 分钟)
# ============================================================================
SIMULATION_START = 0        # 11:30 对应 0 分钟
SIMULATION_END = 60         # 12:30 对应 60 分钟
EXTENDED_END = 70           # 额外10分钟清理积压订单
TIME_STEP = 1               # 时间步长: 1分钟
RHC_INTERVAL = 5            # Rolling Horizon Control 调度间隔: 5分钟

# ============================================================================
# Graph/Map Node Definitions
# ============================================================================
# 节点定义
NODE_HUB = "H"              # Hub (配送中心)
NODE_RESTAURANTS = ["R1", "R2", "R3", "R4"]  # 餐厅节点
NODE_DORMITORIES = ["D1", "D2", "D3", "D4"]  # 宿舍节点
ALL_NODES = [NODE_HUB] + NODE_RESTAURANTS + NODE_DORMITORIES

# ============================================================================
# Base Travel Times (Minutes) - 基础行程时间矩阵
# ============================================================================
# 格式: (origin, destination) -> base_time
# H -> R: Hub到各餐厅的基础时间
# R -> D: 餐厅到各宿舍的基础时间
# R -> H: 返回Hub的时间

BASE_TRAVEL_TIME = {
    # Hub (H) <-> Restaurants (R)
    ("H", "R1"): 3, ("R1", "H"): 3,
    ("H", "R2"): 4, ("R2", "H"): 4,
    ("H", "R3"): 5, ("R3", "H"): 5,
    ("H", "R4"): 6, ("R4", "H"): 6,
    
    # Restaurants (R) <-> Dormitories (D) - 完整矩阵
    ("R1", "D1"): 5, ("D1", "R1"): 5,
    ("R1", "D2"): 6, ("D2", "R1"): 6,
    ("R1", "D3"): 7, ("D3", "R1"): 7,
    ("R1", "D4"): 8, ("D4", "R1"): 8,
    
    ("R2", "D1"): 4, ("D1", "R2"): 4,
    ("R2", "D2"): 5, ("D2", "R2"): 5,
    ("R2", "D3"): 6, ("D3", "R2"): 6,
    ("R2", "D4"): 7, ("D4", "R2"): 7,
    
    ("R3", "D1"): 6, ("D1", "R3"): 6,
    ("R3", "D2"): 5, ("D2", "R3"): 5,
    ("R3", "D3"): 4, ("D3", "R3"): 4,
    ("R3", "D4"): 5, ("D4", "R3"): 5,
    
    ("R4", "D1"): 7, ("D1", "R4"): 7,
    ("R4", "D2"): 6, ("D2", "R4"): 6,
    ("R4", "D3"): 5, ("D3", "R4"): 5,
    ("R4", "D4"): 4, ("D4", "R4"): 4,
    
    # Hub (H) <-> Dormitories (D) - 直达距离
    ("H", "D1"): 8, ("D1", "H"): 8,
    ("H", "D2"): 9, ("D2", "H"): 9,
    ("H", "D3"): 10, ("D3", "H"): 10,
    ("H", "D4"): 11, ("D4", "H"): 11,
}

# ============================================================================
# No-Fly Zone Constraints (无人机禁飞区)
# ============================================================================
# 无人机禁止飞行的路径: R2 <-> D3
DRONE_NO_FLY_EDGES = [
    ("R2", "D3"), ("D3", "R2")
]

# ============================================================================
# Fleet Configuration (异构车队配置)
# ============================================================================

# Agent Types
AGENT_TYPE_HUMAN = "Human"
AGENT_TYPE_AGV = "AGV"
AGENT_TYPE_DRONE = "Drone"

FLEET_CONFIG = {
    AGENT_TYPE_HUMAN: {
        "count": 20,        # 人类骑手数量
        "capacity": 5,      # 容量 (支持批量配送)
        "speed_multiplier": 1.0,  # 速度倍数 (基准)
        "cost_per_order": 1.0,    # 每订单成本
    },
    AGENT_TYPE_AGV: {
        "count": 10,        # AGV数量
        "capacity": 1,      # 容量
        "speed_multiplier": 0.8,  # 速度倍数 (比人快, time = 0.8 * base)
        "cost_per_order": 0.6,    # 每订单成本
    },
    AGENT_TYPE_DRONE: {
        "count": 6,         # 无人机数量
        "capacity": 1,      # 容量
        "speed_multiplier": 0.5,  # 速度倍数 (最快, time = 0.5 * base)
        "cost_per_order": 0.4,    # 每订单成本
    },
}

# ============================================================================
# Demand Constraints (需求约束)
# ============================================================================
HARD_DEADLINE = 30  # 硬性截止时间: 订单生成后30分钟内必须送达

# ============================================================================
# NHPP Demand Generation Parameters (非齐次泊松过程需求生成参数)
# ============================================================================
# 每10分钟区间的订单数量
DEMAND_INTERVALS = [
    (0, 10, 55),   # 11:30-11:40: 55 orders
    (10, 20, 75),  # 11:40-11:50: 75 orders
    (20, 30, 98),  # 11:50-12:00: 98 orders
    (30, 40, 107), # 12:00-12:10: 107 orders (Peak)
    (40, 50, 84),  # 12:10-12:20: 84 orders
    (50, 60, 64),  # 12:20-12:30: 64 orders
]

TOTAL_ORDERS = sum([interval[2] for interval in DEMAND_INTERVALS])

# ============================================================================
# Spatial Distribution (空间分布)
# ============================================================================
# 餐厅订单分布概率
RESTAURANT_DISTRIBUTION = {
    "R1": 0.30,  # 30%
    "R2": 0.25,  # 25%
    "R3": 0.25,  # 25%
    "R4": 0.20,  # 20%
}

# 宿舍配送分布概率
DORMITORY_DISTRIBUTION = {
    "D1": 0.25,  # 25%
    "D2": 0.25,  # 25%
    "D3": 0.25,  # 25%
    "D4": 0.25,  # 25%
}

# ============================================================================
# Dispatcher Parameters (调度参数)
# ============================================================================
CRITICAL_TIME_THRESHOLD = 15  # 紧急订单阈值: 剩余时间 < 15分钟
BUNDLE_DEADLINE_PROXIMITY = 15 # 放宽到15分钟，大幅增加拼单率
LARGE_BUNDLE_THRESHOLD = 3    # 大批量阈值: >= 3个订单


def get_travel_time(origin: str, destination: str, agent_type: str) -> float:
    """
    计算从起点到终点的行程时间，考虑代理类型的速度倍数。
    
    Args:
        origin: 起点节点
        destination: 终点节点
        agent_type: 代理类型 (Human/AGV/Drone)
    
    Returns:
        考虑速度倍数后的行程时间（分钟）
    """
    base_time = BASE_TRAVEL_TIME.get((origin, destination))
    if base_time is None:
        # 如果找不到直接路径，返回一个较大的默认值
        return float('inf')
    
    speed_multiplier = FLEET_CONFIG[agent_type]["speed_multiplier"]
    return base_time * speed_multiplier


def is_drone_path_valid(origin: str, destination: str) -> bool:
    """
    检查无人机是否可以飞行该路径（是否在禁飞区内）。
    
    Args:
        origin: 起点节点
        destination: 终点节点
    
    Returns:
        True if path is valid for drones, False otherwise
    """
    return (origin, destination) not in DRONE_NO_FLY_EDGES

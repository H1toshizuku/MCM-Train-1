"""
entities.py - 实体类定义

This module defines the core entity classes for the simulation:
- Order: 订单实体
- Agent: 配送代理基类
- Rider: 人类骑手
- Drone: 无人机
- AGV: 自动导引车
- Bundle: 订单批次

Mathematical Modeling Contest - Problem B: Campus Delivery System Design
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import config


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"         # 待分配
    ASSIGNED = "assigned"       # 已分配，等待取货
    DELIVERING = "delivering"   # 配送中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"           # 配送失败（超时）


class AgentStatus(Enum):
    """代理状态枚举"""
    IDLE = "idle"               # 空闲
    BUSY = "busy"               # 忙碌中


@dataclass
class Order:
    """
    订单实体类
    
    Attributes:
        order_id: 订单唯一标识
        gen_time: 订单生成时间 (分钟)
        restaurant: 取货餐厅 (R1-R4)
        dormitory: 配送目的地宿舍 (D1-D4)
        deadline: 配送截止时间 = gen_time + HARD_DEADLINE
        status: 订单当前状态
        assigned_agent_id: 分配的配送代理ID
        pickup_time: 取货时间
        delivery_time: 实际送达时间
    """
    order_id: int
    gen_time: float
    restaurant: str
    dormitory: str
    deadline: float = field(init=False)
    status: OrderStatus = OrderStatus.PENDING
    assigned_agent_id: Optional[int] = None
    pickup_time: Optional[float] = None
    delivery_time: Optional[float] = None
    
    def __post_init__(self):
        """计算订单截止时间"""
        self.deadline = self.gen_time + config.HARD_DEADLINE
    
    def get_remaining_time(self, current_time: float) -> float:
        """
        计算剩余配送时间
        
        Args:
            current_time: 当前仿真时间
        
        Returns:
            剩余时间（分钟），可能为负（已超时）
        """
        return self.deadline - current_time
    
    def is_critical(self, current_time: float) -> bool:
        """
        判断是否为紧急订单
        
        Urgency判定: remaining_time < CRITICAL_TIME_THRESHOLD (15分钟)
        """
        return self.get_remaining_time(current_time) < config.CRITICAL_TIME_THRESHOLD
    
    def mark_assigned(self, agent_id: int, pickup_time: float):
        """标记订单为已分配状态"""
        self.status = OrderStatus.ASSIGNED
        self.assigned_agent_id = agent_id
        self.pickup_time = pickup_time
    
    def mark_delivering(self):
        """标记订单为配送中状态"""
        self.status = OrderStatus.DELIVERING
    
    def mark_completed(self, delivery_time: float):
        """标记订单为已完成状态，检查是否超时"""
        self.delivery_time = delivery_time
        if delivery_time <= self.deadline:
            self.status = OrderStatus.COMPLETED
        else:
            self.status = OrderStatus.FAILED
    
    def is_on_time(self) -> bool:
        """判断订单是否按时送达"""
        return self.status == OrderStatus.COMPLETED
    
    def get_delay(self) -> float:
        """获取延迟时间（如果有）"""
        if self.delivery_time is None:
            return 0.0
        return max(0, self.delivery_time - self.deadline)
    
    def __repr__(self):
        return (f"Order(id={self.order_id}, gen={self.gen_time:.1f}, "
                f"{self.restaurant}->{self.dormitory}, status={self.status.value})")


@dataclass
class Agent:
    """
    配送代理基类
    
    Attributes:
        agent_id: 代理唯一标识
        agent_type: 代理类型 (Human/AGV/Drone)
        capacity: 最大容量
        speed_multiplier: 速度倍数
        cost_per_order: 每订单成本
        location: 当前位置
        next_free_time: 下次空闲时间
        status: 当前状态
        current_load: 当前负载（订单数）
        total_orders_delivered: 累计配送订单数
        total_distance_traveled: 累计行驶距离
    """
    agent_id: int
    agent_type: str
    capacity: int = field(init=False)
    speed_multiplier: float = field(init=False)
    cost_per_order: float = field(init=False)
    location: str = config.NODE_HUB  # 初始位置在Hub
    next_free_time: float = 0.0
    status: AgentStatus = AgentStatus.IDLE
    current_load: int = 0
    total_orders_delivered: int = 0
    total_distance_traveled: float = 0.0
    
    def __post_init__(self):
        """根据代理类型初始化配置参数"""
        fleet_config = config.FLEET_CONFIG[self.agent_type]
        self.capacity = fleet_config["capacity"]
        self.speed_multiplier = fleet_config["speed_multiplier"]
        self.cost_per_order = fleet_config["cost_per_order"]
    
    def is_available(self, current_time: float) -> bool:
        """
        检查代理是否可用
        
        Agent is available if:
        1. current_time >= next_free_time
        2. status is IDLE
        
        Args:
            current_time: 当前仿真时间
        
        Returns:
            True if available, False otherwise
        """
        return current_time >= self.next_free_time and self.status == AgentStatus.IDLE
    
    def assign_task(self, task_duration: float, current_time: float, 
                    order_count: int = 1, start_location: str = None, 
                    end_location: str = None):
        """
        分配任务给代理
        
        使用往返逻辑 (Round Trip): 锁定代理 2 * travel_time
        
        Args:
            task_duration: 任务持续时间（往返总时间）
            current_time: 当前时间
            order_count: 订单数量
            start_location: 起始位置
            end_location: 结束位置
        """
        self.status = AgentStatus.BUSY
        self.next_free_time = current_time + task_duration
        self.current_load = order_count
        self.total_orders_delivered += order_count
        
        if end_location:
            # 任务完成后代理返回Hub，但记录终点位置
            self.location = config.NODE_HUB  # 往返后回到Hub
    
    def complete_task(self):
        """完成任务，代理变为空闲"""
        self.status = AgentStatus.IDLE
        self.current_load = 0
        self.location = config.NODE_HUB
    
    def get_travel_time(self, origin: str, destination: str) -> float:
        """
        计算该代理从origin到destination的行程时间
        
        Args:
            origin: 起点
            destination: 终点
        
        Returns:
            考虑速度倍数的行程时间
        """
        return config.get_travel_time(origin, destination, self.agent_type)
    
    def calculate_delivery_time(self, restaurant: str, dormitory: str) -> float:
        """
        计算完成一个配送任务的总时间（往返）
        
        Route: Hub -> Restaurant -> Dormitory -> Hub
        Total time = (H->R + R->D + D->H) * speed_multiplier
        
        简化模型：使用往返时间 = 2 * (H->R + R->D)
        
        Args:
            restaurant: 餐厅位置
            dormitory: 宿舍位置
        
        Returns:
            往返总时间
        """
        time_to_restaurant = self.get_travel_time(config.NODE_HUB, restaurant)
        time_to_dormitory = self.get_travel_time(restaurant, dormitory)
        time_return = self.get_travel_time(dormitory, config.NODE_HUB)
        
        return time_to_restaurant + time_to_dormitory + time_return
    
    def can_handle_route(self, restaurant: str, dormitory: str) -> bool:
        """
        检查代理是否可以处理该路径
        
        对于无人机，需要检查禁飞区
        
        Args:
            restaurant: 餐厅位置
            dormitory: 宿舍位置
        
        Returns:
            True if route is valid for this agent
        """
        return True  # 默认允许，子类可override
    
    def __repr__(self):
        return (f"{self.agent_type}(id={self.agent_id}, "
                f"status={self.status.value}, "
                f"free_at={self.next_free_time:.1f})")


class Rider(Agent):
    """
    人类骑手类
    
    - 容量: 5 (支持批量配送)
    - 速度倍数: 1.0 (基准速度)
    - 成本: 1.0/订单
    """
    
    def __init__(self, agent_id: int):
        super().__init__(agent_id=agent_id, agent_type=config.AGENT_TYPE_HUMAN)


class AGV(Agent):
    """
    自动导引车类
    
    - 容量: 1
    - 速度倍数: 0.8 (比人快)
    - 成本: 0.6/订单
    """
    
    def __init__(self, agent_id: int):
        super().__init__(agent_id=agent_id, agent_type=config.AGENT_TYPE_AGV)


class Drone(Agent):
    """
    无人机类
    
    - 容量: 1
    - 速度倍数: 0.5 (最快)
    - 成本: 0.4/订单
    - 约束: R2 <-> D3 禁飞
    """
    
    def __init__(self, agent_id: int):
        super().__init__(agent_id=agent_id, agent_type=config.AGENT_TYPE_DRONE)
    
    def can_handle_route(self, restaurant: str, dormitory: str) -> bool:
        """
        检查无人机是否可以飞行该路径
        
        禁飞区: R2 <-> D3
        
        Args:
            restaurant: 餐厅位置
            dormitory: 宿舍位置
        
        Returns:
            True if route is valid (not in no-fly zone)
        """
        # 检查 Hub -> Restaurant 是否禁飞
        if not config.is_drone_path_valid(config.NODE_HUB, restaurant):
            return False
        # 检查 Restaurant -> Dormitory 是否禁飞
        if not config.is_drone_path_valid(restaurant, dormitory):
            return False
        # 检查 Dormitory -> Hub 返程是否禁飞
        if not config.is_drone_path_valid(dormitory, config.NODE_HUB):
            return False
        return True


@dataclass
class Bundle:
    """
    订单批次类（用于人类骑手的批量配送）
    
    Attributes:
        bundle_id: 批次唯一标识
        orders: 批次中的订单列表
        restaurant: 取货餐厅（批次内订单同一餐厅）
        dormitories: 配送目的地列表
        earliest_deadline: 批次中最早的截止时间
    """
    bundle_id: int
    orders: List[Order]
    restaurant: str = field(init=False)
    dormitories: List[str] = field(init=False)
    earliest_deadline: float = field(init=False)
    
    def __post_init__(self):
        if self.orders:
            self.restaurant = self.orders[0].restaurant
            self.dormitories = [o.dormitory for o in self.orders]
            self.earliest_deadline = min(o.deadline for o in self.orders)
        else:
            self.restaurant = ""
            self.dormitories = []
            self.earliest_deadline = float('inf')
    
    @property
    def size(self) -> int:
        """批次大小"""
        return len(self.orders)
    
    def get_order_ids(self) -> List[int]:
        """获取所有订单ID"""
        return [o.order_id for o in self.orders]
    
    def get_remaining_time(self, current_time: float) -> float:
        """获取批次剩余时间（以最早截止时间为准）"""
        return self.earliest_deadline - current_time
    
    def is_critical(self, current_time: float) -> bool:
        """判断批次是否紧急"""
        return self.get_remaining_time(current_time) < config.CRITICAL_TIME_THRESHOLD
    
    def __repr__(self):
        return (f"Bundle(id={self.bundle_id}, size={self.size}, "
                f"restaurant={self.restaurant}, deadline={self.earliest_deadline:.1f})")

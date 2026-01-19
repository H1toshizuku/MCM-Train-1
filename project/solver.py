"""
solver.py - 调度算法模块

This module implements the Two-Stage Heuristic dispatcher:
- Stage 1: Batching (Clustering) - 订单批次化
- Stage 2: Assignment (Greedy Utility) - 贪婪分配

Mathematical Modeling Contest - Problem B: Campus Delivery System Design

Dispatching Logic:
1. 按餐厅分组订单
2. 按截止时间接近度划分子批次
3. 根据紧急程度和批次大小分配最优代理
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import config
from entities import Order, Agent, Bundle, OrderStatus


class Dispatcher:
    """
    调度器类
    
    实现两阶段启发式算法:
    - 阶段1: 批次化 (Clustering/Batching)
    - 阶段2: 分配 (Greedy Utility Assignment)
    """
    
    def __init__(self):
        """初始化调度器"""
        self.bundle_counter = 0
        self.dispatch_log = []  # 调度日志
    
    def dispatch(self, current_time: float, pending_orders: List[Order],
                 available_agents: Dict[str, List[Agent]]) -> List[Tuple[Bundle, Agent]]:
        """
        执行调度算法
        
        Args:
            current_time: 当前仿真时间
            pending_orders: 待分配订单列表
            available_agents: 可用代理字典 {type: [agents]}
        
        Returns:
            分配结果列表 [(Bundle, Agent), ...]
        """
        if not pending_orders:
            return []
        
        assignments = []
        
        # Stage 1: 批次化
        bundles = self._create_bundles(pending_orders, current_time)
        
        # Stage 2: 按紧急程度排序
        bundles.sort(key=lambda b: b.get_remaining_time(current_time))
        
        # Stage 2: 贪婪分配
        for bundle in bundles:
            agent = self._assign_bundle(bundle, current_time, available_agents)
            if agent is not None:
                assignments.append((bundle, agent))
                
                # 记录调度日志
                self.dispatch_log.append({
                    "time": current_time,
                    "bundle_id": bundle.bundle_id,
                    "orders": bundle.get_order_ids(),
                    "agent_type": agent.agent_type,
                    "agent_id": agent.agent_id,
                })
        
        return assignments
    
    def _create_bundles(self, orders: List[Order], current_time: float) -> List[Bundle]:
        """
        Stage 1: 批次化 (Clustering)
        
        Algorithm:
        1. 按餐厅分组订单
        2. 在每个餐厅组内，按截止时间接近度划分子批次
        3. 每个子批次大小不超过 Human Rider 容量 (5)
        
        Batching Criteria:
        - 同一餐厅 (Same restaurant)
        - 截止时间在 BUNDLE_DEADLINE_PROXIMITY (5分钟) 内
        - 批次大小 ≤ capacity (5)
        
        Args:
            orders: 待批次化的订单列表
            current_time: 当前仿真时间
        
        Returns:
            批次列表
        """
        bundles = []
        
        # 按餐厅分组
        orders_by_restaurant = defaultdict(list)
        for order in orders:
            orders_by_restaurant[order.restaurant].append(order)
        
        # 对每个餐厅的订单进行批次划分
        for restaurant, restaurant_orders in orders_by_restaurant.items():
            # 按截止时间排序
            restaurant_orders.sort(key=lambda o: o.deadline)
            
            # 划分子批次
            current_batch = []
            batch_earliest_deadline = None
            
            for order in restaurant_orders:
                if not current_batch:
                    # 开始新批次
                    current_batch = [order]
                    batch_earliest_deadline = order.deadline
                else:
                    # 检查是否可以加入当前批次
                    deadline_diff = abs(order.deadline - batch_earliest_deadline)
                    can_add = (
                        len(current_batch) < config.FLEET_CONFIG[config.AGENT_TYPE_HUMAN]["capacity"]
                        and deadline_diff <= config.BUNDLE_DEADLINE_PROXIMITY
                    )
                    
                    if can_add:
                        current_batch.append(order)
                        # 更新最早截止时间
                        batch_earliest_deadline = min(batch_earliest_deadline, order.deadline)
                    else:
                        # 保存当前批次，开始新批次
                        bundle = Bundle(
                            bundle_id=self.bundle_counter,
                            orders=current_batch
                        )
                        bundles.append(bundle)
                        self.bundle_counter += 1
                        
                        current_batch = [order]
                        batch_earliest_deadline = order.deadline
            
            # 保存最后一个批次
            if current_batch:
                bundle = Bundle(
                    bundle_id=self.bundle_counter,
                    orders=current_batch
                )
                bundles.append(bundle)
                self.bundle_counter += 1
        
        return bundles
    
    def _assign_bundle(self, bundle: Bundle, current_time: float,
                       available_agents: Dict[str, List[Agent]]) -> Optional[Agent]:
        """
        Stage 2: 分配批次到代理
        
        Utility Function / Assignment Rules:
        1. Critical (remaining < 15min) & Single: Drone (if valid) > AGV > Human
        2. Large Bundles (size >= 3): Human Rider only
        3. Small Bundles/Single: AGV > Human (fallback)
        
        Constraints:
        - Drone 不能飞行 R2 <-> D3 (No-Fly Zone)
        - 代理容量限制
        
        Args:
            bundle: 待分配的批次
            current_time: 当前仿真时间
            available_agents: 可用代理字典
        
        Returns:
            分配的代理，如果无可用代理则返回 None
        """
        is_critical = bundle.is_critical(current_time)
        is_single = bundle.size == 1
        is_large = bundle.size >= config.LARGE_BUNDLE_THRESHOLD
        
        # 获取路径信息（用于判断无人机禁飞区）
        restaurant = bundle.restaurant
        # 对于批量配送，取第一个目的地判断
        primary_dormitory = bundle.dormitories[0] if bundle.dormitories else None
        
        selected_agent = None
        
        # Rule 1: Critical & Single -> Drone > AGV > Human
        if is_critical and is_single:
            # 尝试无人机
            selected_agent = self._find_best_drone(
                restaurant, primary_dormitory, available_agents
            )
            
            # 尝试 AGV
            if selected_agent is None:
                selected_agent = self._find_best_agv(
                    available_agents, bundle.size
                )
            
            # 最后尝试 Human
            if selected_agent is None:
                selected_agent = self._find_best_human(
                    available_agents, bundle.size
                )
        
        # Rule 2: Large Bundles -> Human Rider only
        elif is_large:
            selected_agent = self._find_best_human(
                available_agents, bundle.size
            )
        
        # Rule 3: Small Bundles/Single -> AGV > Human
        else:
            # 如果是单个订单，也可以考虑无人机
            if is_single:
                selected_agent = self._find_best_drone(
                    restaurant, primary_dormitory, available_agents
                )
            
            # 尝试 AGV
            if selected_agent is None:
                selected_agent = self._find_best_agv(
                    available_agents, bundle.size
                )
            
            # 最后尝试 Human
            if selected_agent is None:
                selected_agent = self._find_best_human(
                    available_agents, bundle.size
                )
        
        return selected_agent
    
    def _find_best_drone(self, restaurant: str, dormitory: str,
                         available_agents: Dict[str, List[Agent]]) -> Optional[Agent]:
        """
        找到可用的最佳无人机
        
        Constraints:
        - 检查禁飞区 (R2 <-> D3)
        - 容量限制 (只能处理单个订单)
        
        Args:
            restaurant: 餐厅位置
            dormitory: 宿舍位置
            available_agents: 可用代理字典
        
        Returns:
            最佳无人机或 None
        """
        drones = available_agents.get(config.AGENT_TYPE_DRONE, [])
        
        for drone in drones:
            # 检查路径是否有效（禁飞区）
            if drone.can_handle_route(restaurant, dormitory):
                # 从可用列表中移除
                available_agents[config.AGENT_TYPE_DRONE].remove(drone)
                return drone
        
        return None
    
    def _find_best_agv(self, available_agents: Dict[str, List[Agent]],
                       required_capacity: int) -> Optional[Agent]:
        """
        找到可用的最佳 AGV
        
        Constraints:
        - 容量限制 (AGV capacity = 1)
        
        Args:
            available_agents: 可用代理字典
            required_capacity: 需要的容量
        
        Returns:
            最佳 AGV 或 None
        """
        if required_capacity > config.FLEET_CONFIG[config.AGENT_TYPE_AGV]["capacity"]:
            return None
        
        agvs = available_agents.get(config.AGENT_TYPE_AGV, [])
        
        if agvs:
            agv = agvs.pop(0)  # 取第一个可用的
            return agv
        
        return None
    
    def _find_best_human(self, available_agents: Dict[str, List[Agent]],
                         required_capacity: int) -> Optional[Agent]:
        """
        找到可用的最佳人类骑手
        
        Constraints:
        - 容量限制 (Human capacity = 5)
        
        Args:
            available_agents: 可用代理字典
            required_capacity: 需要的容量
        
        Returns:
            最佳人类骑手或 None
        """
        if required_capacity > config.FLEET_CONFIG[config.AGENT_TYPE_HUMAN]["capacity"]:
            return None
        
        humans = available_agents.get(config.AGENT_TYPE_HUMAN, [])
        
        if humans:
            human = humans.pop(0)  # 取第一个可用的
            return human
        
        return None
    
    def calculate_assignment_cost(self, bundle: Bundle, agent: Agent) -> float:
        """
        计算分配成本
        
        Cost = agent.cost_per_order * bundle.size
        
        Args:
            bundle: 订单批次
            agent: 分配的代理
        
        Returns:
            总成本
        """
        return agent.cost_per_order * bundle.size
    
    def calculate_delivery_time(self, bundle: Bundle, agent: Agent,
                                current_time: float) -> float:
        """
        计算配送完成时间
        
        Simple Model (Single destination):
        delivery_time = current_time + travel_time(H->R) + travel_time(R->D)
        
        For bundles with multiple destinations:
        Use the furthest dormitory as the completion time estimate
        
        Args:
            bundle: 订单批次
            agent: 分配的代理
            current_time: 当前时间
        
        Returns:
            预计送达时间
        """
        restaurant = bundle.restaurant
        
        # 取货时间
        time_to_restaurant = agent.get_travel_time(config.NODE_HUB, restaurant)
        pickup_time = current_time + time_to_restaurant
        
        # 配送时间（取所有目的地中最长的）
        max_delivery_time = 0
        for dormitory in bundle.dormitories:
            delivery_duration = agent.get_travel_time(restaurant, dormitory)
            max_delivery_time = max(max_delivery_time, delivery_duration)
        
        return pickup_time + max_delivery_time
    
    def get_dispatch_summary(self) -> Dict:
        """
        获取调度统计摘要
        
        Returns:
            包含调度统计信息的字典
        """
        summary = {
            "total_dispatches": len(self.dispatch_log),
            "dispatches_by_agent_type": defaultdict(int),
            "orders_by_agent_type": defaultdict(int),
        }
        
        for log in self.dispatch_log:
            agent_type = log["agent_type"]
            summary["dispatches_by_agent_type"][agent_type] += 1
            summary["orders_by_agent_type"][agent_type] += len(log["orders"])
        
        return summary


if __name__ == "__main__":
    # 测试调度器
    from generator import DemandGenerator
    from entities import Rider, AGV, Drone
    
    # 生成订单
    generator = DemandGenerator()
    orders = generator.generate_orders()
    
    # 创建代理
    agents = {
        config.AGENT_TYPE_HUMAN: [Rider(i) for i in range(5)],
        config.AGENT_TYPE_AGV: [AGV(i) for i in range(3)],
        config.AGENT_TYPE_DRONE: [Drone(i) for i in range(2)],
    }
    
    # 测试调度
    dispatcher = Dispatcher()
    
    # 取前20个订单进行测试
    test_orders = orders[:20]
    current_time = 5.0
    
    # 只取待处理的订单
    pending = [o for o in test_orders if o.status == OrderStatus.PENDING]
    
    # 获取可用代理
    available = {
        agent_type: [a for a in agent_list if a.is_available(current_time)]
        for agent_type, agent_list in agents.items()
    }
    
    # 执行调度
    assignments = dispatcher.dispatch(current_time, pending, available)
    
    print(f"调度了 {len(assignments)} 个批次")
    for bundle, agent in assignments:
        print(f"  {bundle} -> {agent}")
    
    # 打印调度摘要
    summary = dispatcher.get_dispatch_summary()
    print(f"\n调度摘要:")
    print(f"  总调度次数: {summary['total_dispatches']}")
    print(f"  按代理类型分布: {dict(summary['dispatches_by_agent_type'])}")
    print(f"  订单分布: {dict(summary['orders_by_agent_type'])}")

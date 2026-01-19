"""
simulator.py - 离散事件仿真主模块

This module implements the main Discrete Event Simulation (DES) loop
with Rolling Horizon Control (RHC) framework.

Mathematical Modeling Contest - Problem B: Campus Delivery System Design

Simulation Logic (Rolling Horizon Control):
1. Reveal: 添加新生成的订单到待处理池
2. Check: 更新代理状态（释放完成任务的代理）
3. Dispatch: 每5分钟触发RHC调度
4. Update: 分配任务，更新订单状态
5. Step: 时间步进
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import config
from entities import (
    Order, Agent, Rider, AGV, Drone, Bundle,
    OrderStatus, AgentStatus
)
from generator import DemandGenerator
from solver import Dispatcher


@dataclass
class SimulationEvent:
    """仿真事件类"""
    time: float
    event_type: str  # "order_reveal", "agent_free", "dispatch", "delivery_complete"
    data: dict = field(default_factory=dict)


class Environment:
    """
    仿真环境类
    
    管理整个仿真过程，包括:
    - 订单池管理
    - 代理状态管理
    - 时间步进
    - 事件调度
    """
    
    def __init__(self, orders: List[Order] = None, verbose: bool = True):
        """
        初始化仿真环境
        
        Args:
            orders: 预生成的订单列表，如果为None则自动生成
            verbose: 是否打印详细日志
        """
        self.verbose = verbose
        
        # 订单管理
        if orders is None:
            generator = DemandGenerator()
            self.all_orders = generator.generate_orders()
        else:
            self.all_orders = orders
        
        self.pending_pool: List[Order] = []      # 待分配订单池
        self.assigned_orders: List[Order] = []   # 已分配订单
        self.completed_orders: List[Order] = []  # 已完成订单
        self.failed_orders: List[Order] = []     # 失败订单
        
        # 订单索引（按时间排序后的位置）
        self.order_reveal_index = 0
        
        # 代理管理
        self.agents = self._initialize_fleet()
        
        # 调度器
        self.dispatcher = Dispatcher()
        
        # 仿真状态
        self.current_time = 0.0
        self.time_step = config.TIME_STEP
        
        # 统计数据
        self.statistics = SimulationStatistics()
        
        # 事件日志
        self.event_log = []
        
    def _initialize_fleet(self) -> Dict[str, List[Agent]]:
        """
        初始化异构车队
        
        Returns:
            代理字典 {type: [agents]}
        """
        fleet = {
            config.AGENT_TYPE_HUMAN: [],
            config.AGENT_TYPE_AGV: [],
            config.AGENT_TYPE_DRONE: [],
        }
        
        # 创建人类骑手
        for i in range(config.FLEET_CONFIG[config.AGENT_TYPE_HUMAN]["count"]):
            fleet[config.AGENT_TYPE_HUMAN].append(Rider(agent_id=i))
        
        # 创建AGV
        for i in range(config.FLEET_CONFIG[config.AGENT_TYPE_AGV]["count"]):
            fleet[config.AGENT_TYPE_AGV].append(AGV(agent_id=i))
        
        # 创建无人机
        for i in range(config.FLEET_CONFIG[config.AGENT_TYPE_DRONE]["count"]):
            fleet[config.AGENT_TYPE_DRONE].append(Drone(agent_id=i))
        
        if self.verbose:
            total_agents = sum(len(agents) for agents in fleet.values())
            print(f"初始化车队: 共 {total_agents} 个代理")
            for agent_type, agents in fleet.items():
                print(f"  {agent_type}: {len(agents)}")
        
        return fleet
    
    def run(self) -> Dict:
        """
        运行仿真
        
        Main Loop:
        1. Reveal: 添加新订单
        2. Check: 更新代理状态
        3. Dispatch: 每5分钟触发调度
        4. Update: 处理分配
        5. Step: 时间步进
        
        Returns:
            仿真结果统计
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("开始仿真")
            print(f"仿真时间: {config.SIMULATION_START} - {config.EXTENDED_END} 分钟")
            print(f"总订单数: {len(self.all_orders)}")
            print("=" * 60)
        
        # 主循环
        while self.current_time <= config.EXTENDED_END:
            # 1. Reveal: 添加新生成的订单
            self._reveal_orders()
            
            # 2. Check: 更新代理状态
            self._update_agent_status()
            
            # 3. Check: 处理配送完成
            self._process_deliveries()
            
            # 4. Dispatch: 调度策略
            # 4a. 定期RHC调度 (每 RHC_INTERVAL 分钟)
            if self.current_time % config.RHC_INTERVAL == 0:
                self._dispatch()
            # 4b. 紧急调度: 有紧急订单且有可用代理时立即调度
            elif self._has_critical_orders() and self._has_available_agents():
                self._dispatch_urgent()
            
            # 5. 记录统计
            self._record_statistics()
            
            # 6. Step: 时间步进
            self.current_time += self.time_step
        
        # 处理剩余未完成订单
        self._finalize()
        
        # 返回结果
        return self._get_results()
    
    def _reveal_orders(self):
        """
        Reveal阶段: 将新生成的订单添加到待处理池
        
        订单按生成时间排序，只添加 gen_time <= current_time 的订单
        """
        while (self.order_reveal_index < len(self.all_orders) and
               self.all_orders[self.order_reveal_index].gen_time <= self.current_time):
            
            order = self.all_orders[self.order_reveal_index]
            self.pending_pool.append(order)
            self.order_reveal_index += 1
            
            if self.verbose and self.current_time % 10 == 0:
                pass  # 减少日志输出
    
    def _update_agent_status(self):
        """
        Check阶段: 更新代理状态
        
        释放已完成任务的代理
        """
        for agent_type, agents in self.agents.items():
            for agent in agents:
                if (agent.status == AgentStatus.BUSY and 
                    self.current_time >= agent.next_free_time):
                    agent.complete_task()
    
    def _process_deliveries(self):
        """
        处理配送完成事件
        
        检查已分配订单是否完成配送
        """
        completed = []
        for order in self.assigned_orders:
            if order.status == OrderStatus.DELIVERING:
                # 检查是否已送达（基于分配时计算的送达时间）
                if hasattr(order, '_expected_delivery_time'):
                    if self.current_time >= order._expected_delivery_time:
                        order.mark_completed(order._expected_delivery_time)
                        completed.append(order)
        
        # 移动到完成/失败列表
        for order in completed:
            self.assigned_orders.remove(order)
            if order.status == OrderStatus.COMPLETED:
                self.completed_orders.append(order)
                self.statistics.record_completion(order)
            else:
                self.failed_orders.append(order)
                self.statistics.record_failure(order)
    
    def _dispatch(self):
        """
        Dispatch阶段: 执行调度算法
        
        Rolling Horizon Control (RHC):
        - 每 RHC_INTERVAL 分钟触发一次
        - 调度当前待处理池中的所有订单
        """
        if not self.pending_pool:
            return
        
        if self.verbose:
            print(f"\n[t={self.current_time:.0f}] RHC调度触发, 待处理订单: {len(self.pending_pool)}")
        
        # 获取可用代理
        available_agents = self._get_available_agents()
        
        # 复制待处理订单列表（避免在迭代中修改）
        pending_orders = list(self.pending_pool)
        
        # 执行调度
        assignments = self.dispatcher.dispatch(
            self.current_time, pending_orders, available_agents
        )
        
        # 处理分配结果
        for bundle, agent in assignments:
            self._process_assignment(bundle, agent)
        
        if self.verbose and assignments:
            print(f"  分配了 {len(assignments)} 个批次")
    
    def _get_available_agents(self) -> Dict[str, List[Agent]]:
        """
        获取当前可用的代理
        
        Returns:
            可用代理字典 {type: [available_agents]}
        """
        available = {}
        for agent_type, agents in self.agents.items():
            available[agent_type] = [
                a for a in agents if a.is_available(self.current_time)
            ]
        return available
    
    def _has_critical_orders(self) -> bool:
        """检查是否有紧急订单（剩余时间 < 15分钟）"""
        for order in self.pending_pool:
            if order.is_critical(self.current_time):
                return True
        return False
    
    def _has_available_agents(self) -> bool:
        """检查是否有可用代理"""
        for agent_type, agents in self.agents.items():
            for agent in agents:
                if agent.is_available(self.current_time):
                    return True
        return False
    
    def _dispatch_urgent(self):
        """
        紧急调度：只处理紧急订单
        
        优先使用快速代理(Drone/AGV)处理紧急订单
        """
        # 筛选紧急订单
        urgent_orders = [
            o for o in self.pending_pool 
            if o.is_critical(self.current_time)
        ]
        
        if not urgent_orders:
            return
        
        # 获取可用代理
        available_agents = self._get_available_agents()
        
        # 执行调度（只处理紧急订单）
        assignments = self.dispatcher.dispatch(
            self.current_time, urgent_orders, available_agents
        )
        
        # 处理分配结果
        for bundle, agent in assignments:
            self._process_assignment(bundle, agent)
    
    def _process_assignment(self, bundle: Bundle, agent: Agent):
        """
        处理单个分配
        
        Args:
            bundle: 分配的订单批次
            agent: 分配的代理
        """
        # 计算配送时间
        restaurant = bundle.restaurant
        
        # 取货时间
        time_to_restaurant = agent.get_travel_time(config.NODE_HUB, restaurant)
        pickup_time = self.current_time + time_to_restaurant
        
        # 计算到各宿舍的配送时间
        for order in bundle.orders:
            delivery_duration = agent.get_travel_time(restaurant, order.dormitory)
            expected_delivery = pickup_time + delivery_duration
            
            # 设置期望送达时间（用于后续处理）
            order._expected_delivery_time = expected_delivery
            
            # 更新订单状态
            order.mark_assigned(agent.agent_id, pickup_time)
            order.mark_delivering()
            
            # 从待处理池移除
            if order in self.pending_pool:
                self.pending_pool.remove(order)
            
            # 添加到已分配列表
            self.assigned_orders.append(order)
        
        # 计算任务时间（不含回程，模拟就地接单/顺路接单的高效流转）
        # 移除 return_time，骑手送完后立即具备接单能力
        total_task_time = time_to_restaurant + max(
            agent.get_travel_time(restaurant, d) for d in bundle.dormitories
        )
        
        agent.assign_task(
            task_duration=total_task_time,
            current_time=self.current_time,
            order_count=bundle.size,
            start_location=config.NODE_HUB,
            end_location=bundle.dormitories[-1]
        )
        
        # 记录成本
        cost = self.dispatcher.calculate_assignment_cost(bundle, agent)
        self.statistics.record_cost(cost, agent.agent_type)
        
        if self.verbose:
            print(f"    Bundle {bundle.bundle_id}: {bundle.size} 订单 -> "
                  f"{agent.agent_type}#{agent.agent_id}, "
                  f"预计完成: t={pickup_time + max(agent.get_travel_time(restaurant, d) for d in bundle.dormitories):.1f}")
    
    def _record_statistics(self):
        """记录当前时刻的统计数据"""
        # 记录代理使用情况
        active_counts = {}
        for agent_type, agents in self.agents.items():
            active = sum(1 for a in agents if a.status == AgentStatus.BUSY)
            active_counts[agent_type] = active
        
        self.statistics.record_agent_usage(self.current_time, active_counts)
        
        # 记录订单状态
        self.statistics.record_order_status(
            self.current_time,
            len(self.pending_pool),
            len(self.assigned_orders),
            len(self.completed_orders),
            len(self.failed_orders)
        )
    
    def _finalize(self):
        """
        仿真结束处理
        
        处理所有剩余未完成的订单
        """
        # 检查仍在配送中的订单
        for order in self.assigned_orders:
            if hasattr(order, '_expected_delivery_time'):
                order.mark_completed(order._expected_delivery_time)
                if order.status == OrderStatus.COMPLETED:
                    self.completed_orders.append(order)
                    self.statistics.record_completion(order)
                else:
                    self.failed_orders.append(order)
                    self.statistics.record_failure(order)
        self.assigned_orders.clear()
        
        # 标记未处理的订单为失败
        for order in self.pending_pool:
            order.status = OrderStatus.FAILED
            order.delivery_time = None
            self.failed_orders.append(order)
            self.statistics.record_failure(order)
        self.pending_pool.clear()
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("仿真结束")
            print("=" * 60)
    
    def _get_results(self) -> Dict:
        """
        获取仿真结果
        
        Returns:
            结果字典
        """
        return {
            "all_orders": self.all_orders,
            "completed_orders": self.completed_orders,
            "failed_orders": self.failed_orders,
            "statistics": self.statistics,
            "dispatch_log": self.dispatcher.dispatch_log,
        }


class SimulationStatistics:
    """仿真统计类"""
    
    def __init__(self):
        """初始化统计数据结构"""
        # 订单统计
        self.total_completed = 0
        self.total_failed = 0
        self.on_time_deliveries = 0
        self.late_deliveries = 0
        self.total_delay = 0.0
        
        # 成本统计
        self.total_cost = 0.0
        self.cost_by_agent_type = defaultdict(float)
        
        # 时间序列数据
        self.agent_usage_over_time = []  # [(time, {type: count}), ...]
        self.order_status_over_time = [] # [(time, pending, assigned, completed, failed), ...]
        
        # 订单完成记录
        self.completion_records = []  # [{order_id, gen_time, delivery_time, delay}, ...]
    
    def record_completion(self, order: Order):
        """记录订单完成"""
        self.total_completed += 1
        
        if order.is_on_time():
            self.on_time_deliveries += 1
        else:
            self.late_deliveries += 1
            self.total_delay += order.get_delay()
        
        self.completion_records.append({
            "order_id": order.order_id,
            "gen_time": order.gen_time,
            "delivery_time": order.delivery_time,
            "delay": order.get_delay(),
            "on_time": order.is_on_time(),
        })
    
    def record_failure(self, order: Order):
        """记录订单失败"""
        self.total_failed += 1
        self.late_deliveries += 1
    
    def record_cost(self, cost: float, agent_type: str):
        """记录成本"""
        self.total_cost += cost
        self.cost_by_agent_type[agent_type] += cost
    
    def record_agent_usage(self, time: float, active_counts: Dict[str, int]):
        """记录代理使用情况"""
        self.agent_usage_over_time.append((time, dict(active_counts)))
    
    def record_order_status(self, time: float, pending: int, assigned: int,
                           completed: int, failed: int):
        """记录订单状态"""
        self.order_status_over_time.append((time, pending, assigned, completed, failed))
    
    def get_kpis(self) -> Dict:
        """
        计算关键性能指标 (KPIs)
        
        Returns:
            KPI字典
        """
        total_orders = self.total_completed + self.total_failed
        
        # 准时率
        on_time_rate = (self.on_time_deliveries / total_orders * 100 
                        if total_orders > 0 else 0)
        
        # 平均成本
        avg_cost = (self.total_cost / total_orders 
                    if total_orders > 0 else 0)
        
        # 平均延迟
        avg_delay = (self.total_delay / self.late_deliveries 
                     if self.late_deliveries > 0 else 0)
        
        return {
            "total_orders": total_orders,
            "completed": self.total_completed,
            "failed": self.total_failed,
            "on_time_deliveries": self.on_time_deliveries,
            "late_deliveries": self.late_deliveries,
            "on_time_rate": on_time_rate,
            "total_cost": self.total_cost,
            "avg_cost_per_order": avg_cost,
            "total_delay": self.total_delay,
            "avg_delay": avg_delay,
            "cost_by_agent_type": dict(self.cost_by_agent_type),
        }
    
    def print_summary(self):
        """打印统计摘要"""
        kpis = self.get_kpis()
        
        print("\n" + "=" * 60)
        print("仿真结果摘要")
        print("=" * 60)
        
        print(f"\n订单统计:")
        print(f"  总订单数: {kpis['total_orders']}")
        print(f"  完成订单: {kpis['completed']}")
        print(f"  失败订单: {kpis['failed']}")
        print(f"  准时送达: {kpis['on_time_deliveries']}")
        print(f"  延迟送达: {kpis['late_deliveries']}")
        
        print(f"\n关键性能指标 (KPIs):")
        print(f"  准时率: {kpis['on_time_rate']:.2f}%")
        print(f"  平均每单成本: {kpis['avg_cost_per_order']:.3f}")
        print(f"  平均延迟时间: {kpis['avg_delay']:.2f} 分钟")
        
        print(f"\n成本分布:")
        print(f"  总成本: {kpis['total_cost']:.2f}")
        for agent_type, cost in kpis['cost_by_agent_type'].items():
            pct = cost / kpis['total_cost'] * 100 if kpis['total_cost'] > 0 else 0
            print(f"  {agent_type}: {cost:.2f} ({pct:.1f}%)")
        
        print("=" * 60)


if __name__ == "__main__":
    # 运行仿真测试
    import numpy as np
    np.random.seed(config.RANDOM_SEED)
    
    print("校园配送系统离散事件仿真")
    print("=" * 60)
    
    # 创建并运行仿真
    env = Environment(verbose=True)
    results = env.run()
    
    # 打印结果
    results["statistics"].print_summary()

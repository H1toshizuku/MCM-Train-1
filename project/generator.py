"""
generator.py - 需求生成模块

This module implements the Non-homogeneous Poisson Process (NHPP) 
based demand generation logic with stratified sampling for spatial distribution.

Mathematical Modeling Contest - Problem B: Campus Delivery System Design

NHPP Generation Logic:
- 使用分段常数强度函数 (Piecewise Constant Intensity)
- 每10分钟区间有不同的订单到达率
- 空间分布使用分层抽样确保精确比例
"""

import numpy as np
from typing import List, Tuple
from collections import defaultdict
import config
from entities import Order


class DemandGenerator:
    """
    需求生成器类
    
    使用非齐次泊松过程 (NHPP) 生成订单到达时间，
    并使用分层抽样 (Stratified Sampling) 分配餐厅和宿舍位置。
    """
    
    def __init__(self, seed: int = config.RANDOM_SEED):
        """
        初始化需求生成器
        
        Args:
            seed: 随机数种子，确保可复现性
        """
        self.rng = np.random.RandomState(seed)
        self.order_counter = 0
    
    def generate_orders(self) -> List[Order]:
        """
        生成整个仿真期间的所有订单
        
        NHPP Generation with Piecewise Constant Intensity:
        1. 对每个10分钟区间，生成指定数量的订单
        2. 订单到达时间在区间内均匀分布
        3. 餐厅和宿舍位置使用分层抽样分配
        
        Returns:
            按生成时间排序的订单列表
        """
        all_orders = []
        
        for start_time, end_time, order_count in config.DEMAND_INTERVALS:
            # 在区间内生成订单到达时间
            arrival_times = self._generate_nhpp_arrivals(
                start_time, end_time, order_count
            )
            
            # 为每个订单分配餐厅和宿舍
            restaurants = self._stratified_sample_restaurants(order_count)
            dormitories = self._stratified_sample_dormitories(order_count)
            
            # 创建订单对象
            for gen_time, restaurant, dormitory in zip(
                arrival_times, restaurants, dormitories
            ):
                order = Order(
                    order_id=self.order_counter,
                    gen_time=round(gen_time, 2),  # 保留2位小数
                    restaurant=restaurant,
                    dormitory=dormitory
                )
                all_orders.append(order)
                self.order_counter += 1
        
        # 按生成时间排序
        all_orders.sort(key=lambda x: x.gen_time)
        
        return all_orders
    
    def _generate_nhpp_arrivals(self, start_time: float, end_time: float,
                                 n_orders: int) -> np.ndarray:
        """
        Non-homogeneous Poisson Process (NHPP) 到达时间生成
        
        对于分段常数强度函数，在每个区间内订单均匀到达。
        使用有序统计量方法：生成n个均匀分布的点并排序。
        
        Mathematical Basis:
        - 给定区间 [t_start, t_end] 内有 n 个事件
        - 订单到达时间服从均匀分布
        - 排序后得到有序到达时间
        
        Args:
            start_time: 区间开始时间
            end_time: 区间结束时间
            n_orders: 区间内的订单数量
        
        Returns:
            排序后的到达时间数组
        """
        if n_orders == 0:
            return np.array([])
        
        # 在区间内生成均匀分布的到达时间
        arrival_times = self.rng.uniform(start_time, end_time, n_orders)
        arrival_times.sort()
        
        return arrival_times
    
    def _stratified_sample_restaurants(self, n_orders: int) -> List[str]:
        """
        分层抽样选择餐厅
        
        Stratified Sampling Logic:
        1. 根据分布概率计算每个餐厅的期望订单数
        2. 使用整数部分确保基本份额
        3. 使用随机化处理余数分配
        
        Distribution: R1=30%, R2=25%, R3=25%, R4=20%
        
        Args:
            n_orders: 需要分配的订单总数
        
        Returns:
            餐厅列表，长度为n_orders
        """
        return self._stratified_sample(
            config.RESTAURANT_DISTRIBUTION, n_orders
        )
    
    def _stratified_sample_dormitories(self, n_orders: int) -> List[str]:
        """
        分层抽样选择宿舍
        
        Distribution: D1=25%, D2=25%, D3=25%, D4=25%
        
        Args:
            n_orders: 需要分配的订单总数
        
        Returns:
            宿舍列表，长度为n_orders
        """
        return self._stratified_sample(
            config.DORMITORY_DISTRIBUTION, n_orders
        )
    
    def _stratified_sample(self, distribution: dict, n_samples: int) -> List[str]:
        """
        通用分层抽样方法
        
        Algorithm:
        1. 计算每个类别的期望数量 = probability * n_samples
        2. 取整数部分作为保证配额
        3. 对小数部分使用加权随机抽样分配剩余名额
        
        Args:
            distribution: 类别到概率的映射
            n_samples: 需要抽样的总数
        
        Returns:
            抽样结果列表
        """
        if n_samples == 0:
            return []
        
        categories = list(distribution.keys())
        probabilities = list(distribution.values())
        
        # 计算每个类别的期望数量
        expected_counts = [p * n_samples for p in probabilities]
        
        # 整数部分
        base_counts = [int(c) for c in expected_counts]
        
        # 小数部分（用于分配剩余名额）
        remainders = [c - int(c) for c in expected_counts]
        
        # 剩余需要分配的数量
        remaining = n_samples - sum(base_counts)
        
        # 使用加权随机抽样分配剩余名额
        if remaining > 0:
            # 归一化余数作为权重
            remainder_sum = sum(remainders)
            if remainder_sum > 0:
                weights = [r / remainder_sum for r in remainders]
            else:
                weights = [1.0 / len(categories)] * len(categories)
            
            # 随机选择要增加的类别
            extra_indices = self.rng.choice(
                len(categories), size=remaining, replace=True, p=weights
            )
            for idx in extra_indices:
                base_counts[idx] += 1
        
        # 生成结果列表
        result = []
        for category, count in zip(categories, base_counts):
            result.extend([category] * count)
        
        # 打乱顺序
        self.rng.shuffle(result)
        
        return result


def generate_demand_statistics(orders: List[Order]) -> dict:
    """
    生成需求统计信息
    
    Args:
        orders: 订单列表
    
    Returns:
        包含各种统计数据的字典
    """
    stats = {
        "total_orders": len(orders),
        "orders_by_interval": defaultdict(int),
        "restaurant_distribution": defaultdict(int),
        "dormitory_distribution": defaultdict(int),
        "od_matrix": defaultdict(int),  # Origin-Destination矩阵
    }
    
    for order in orders:
        # 按时间区间统计
        interval_idx = int(order.gen_time // 10) * 10
        stats["orders_by_interval"][f"{interval_idx}-{interval_idx+10}"] += 1
        
        # 餐厅分布
        stats["restaurant_distribution"][order.restaurant] += 1
        
        # 宿舍分布
        stats["dormitory_distribution"][order.dormitory] += 1
        
        # OD矩阵
        stats["od_matrix"][(order.restaurant, order.dormitory)] += 1
    
    return stats


def print_demand_statistics(stats: dict):
    """
    打印需求统计信息
    
    Args:
        stats: 统计数据字典
    """
    print("=" * 60)
    print("需求生成统计报告")
    print("=" * 60)
    
    print(f"\n总订单数: {stats['total_orders']}")
    
    print("\n按时间区间分布:")
    for interval in sorted(stats["orders_by_interval"].keys()):
        count = stats["orders_by_interval"][interval]
        print(f"  {interval}分钟: {count} 订单")
    
    print("\n餐厅分布:")
    total = stats['total_orders']
    for restaurant in sorted(stats["restaurant_distribution"].keys()):
        count = stats["restaurant_distribution"][restaurant]
        pct = count / total * 100
        print(f"  {restaurant}: {count} ({pct:.1f}%)")
    
    print("\n宿舍分布:")
    for dormitory in sorted(stats["dormitory_distribution"].keys()):
        count = stats["dormitory_distribution"][dormitory]
        pct = count / total * 100
        print(f"  {dormitory}: {count} ({pct:.1f}%)")
    
    print("\nOD矩阵 (餐厅 -> 宿舍):")
    print("      ", end="")
    for d in config.NODE_DORMITORIES:
        print(f"{d:>6}", end="")
    print()
    
    for r in config.NODE_RESTAURANTS:
        print(f"  {r}:", end="")
        for d in config.NODE_DORMITORIES:
            count = stats["od_matrix"].get((r, d), 0)
            print(f"{count:>6}", end="")
        print()
    
    print("=" * 60)


if __name__ == "__main__":
    # 测试需求生成
    generator = DemandGenerator()
    orders = generator.generate_orders()
    
    print(f"生成了 {len(orders)} 个订单")
    print("\n前10个订单:")
    for order in orders[:10]:
        print(f"  {order}")
    
    print("\n后10个订单:")
    for order in orders[-10:]:
        print(f"  {order}")
    
    # 统计信息
    stats = generate_demand_statistics(orders)
    print_demand_statistics(stats)

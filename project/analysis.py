"""
analysis.py - 可视化和KPI分析模块

This module implements visualization and KPI calculation functions:
- Demand Heatmap (订单量柱状图)
- Fleet Usage Stacked Area Chart (代理使用堆叠面积图)
- Order Status Pie Chart (订单状态饼图)
- Additional analysis charts

Mathematical Modeling Contest - Problem B: Campus Delivery System Design
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Dict, List, Tuple
from collections import defaultdict
import config
from simulator import SimulationStatistics
from entities import Order


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class SimulationAnalyzer:
    """仿真结果分析器"""
    
    def __init__(self, results: Dict):
        """
        初始化分析器
        
        Args:
            results: 仿真结果字典
        """
        self.all_orders = results.get("all_orders", [])
        self.completed_orders = results.get("completed_orders", [])
        self.failed_orders = results.get("failed_orders", [])
        self.statistics: SimulationStatistics = results.get("statistics")
        self.dispatch_log = results.get("dispatch_log", [])
    
    def generate_all_plots(self, save_path: str = None, show: bool = True):
        """
        生成所有可视化图表
        
        Args:
            save_path: 保存路径（如果指定则保存图片）
            show: 是否显示图表
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("校园配送系统仿真分析报告", fontsize=16, fontweight='bold')
        
        # 1. 需求热力图/柱状图
        self._plot_demand_heatmap(axes[0, 0])
        
        # 2. 车队使用堆叠面积图
        self._plot_fleet_usage(axes[0, 1])
        
        # 3. 订单状态饼图
        self._plot_order_status_pie(axes[1, 0])
        
        # 4. KPI仪表盘
        self._plot_kpi_dashboard(axes[1, 1])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _plot_demand_heatmap(self, ax):
        """
        绘制需求热力图（柱状图）
        
        显示每10分钟区间的订单数量
        """
        # 按时间区间统计订单
        interval_counts = defaultdict(int)
        for order in self.all_orders:
            interval_idx = int(order.gen_time // 10) * 10
            interval_counts[interval_idx] += 1
        
        # 准备数据
        intervals = list(range(0, 60, 10))
        counts = [interval_counts.get(i, 0) for i in intervals]
        labels = [f"{i}-{i+10}" for i in intervals]
        
        # 颜色映射（根据订单量变化）
        colors = plt.cm.YlOrRd(np.array(counts) / max(counts))
        
        bars = ax.bar(labels, counts, color=colors, edgecolor='black', linewidth=0.5)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.annotate(f'{count}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel("时间区间 (分钟)", fontsize=11)
        ax.set_ylabel("订单数量", fontsize=11)
        ax.set_title("订单需求分布 (Demand Heatmap)", fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(counts) * 1.15)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_fleet_usage(self, ax):
        """
        绘制车队使用堆叠面积图
        
        显示每个时间点各类型代理的活跃数量
        """
        usage_data = self.statistics.agent_usage_over_time
        
        if not usage_data:
            ax.text(0.5, 0.5, "无代理使用数据", ha='center', va='center')
            ax.set_title("车队使用情况", fontsize=12, fontweight='bold')
            return
        
        # 准备数据
        times = [d[0] for d in usage_data]
        
        human_usage = [d[1].get(config.AGENT_TYPE_HUMAN, 0) for d in usage_data]
        agv_usage = [d[1].get(config.AGENT_TYPE_AGV, 0) for d in usage_data]
        drone_usage = [d[1].get(config.AGENT_TYPE_DRONE, 0) for d in usage_data]
        
        # 堆叠面积图
        ax.stackplot(times, human_usage, agv_usage, drone_usage,
                    labels=['Human Rider', 'AGV', 'Drone'],
                    colors=['#3498db', '#2ecc71', '#e74c3c'],
                    alpha=0.8)
        
        ax.set_xlabel("仿真时间 (分钟)", fontsize=11)
        ax.set_ylabel("活跃代理数量", fontsize=11)
        ax.set_title("车队使用情况 (Fleet Usage)", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_xlim(0, config.EXTENDED_END)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_order_status_pie(self, ax):
        """
        绘制订单状态饼图
        
        显示准时送达 vs 延迟送达 vs 失败的比例
        """
        kpis = self.statistics.get_kpis()
        
        # 数据
        on_time = kpis['on_time_deliveries']
        late = kpis['late_deliveries']
        failed = kpis['failed']
        
        # 过滤掉0值
        labels = []
        sizes = []
        colors_list = []
        
        color_map = {
            'On-time': '#27ae60',
            'Late': '#f39c12',
            'Failed': '#e74c3c',
        }
        
        if on_time > 0:
            labels.append('On-time')
            sizes.append(on_time)
            colors_list.append(color_map['On-time'])
        if late > 0 and failed == 0:  # 延迟但成功送达
            labels.append('Late')
            sizes.append(late)
            colors_list.append(color_map['Late'])
        if failed > 0:
            labels.append('Failed')
            sizes.append(failed)
            colors_list.append(color_map['Failed'])
        
        if not sizes:
            ax.text(0.5, 0.5, "无订单数据", ha='center', va='center')
            ax.set_title("订单状态分布", fontsize=12, fontweight='bold')
            return
        
        # 饼图
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors_list,
            autopct='%1.1f%%', startangle=90,
            explode=[0.02] * len(sizes),
            shadow=True
        )
        
        # 设置字体
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        ax.set_title("订单状态分布 (Order Status)", fontsize=12, fontweight='bold')
    
    def _plot_kpi_dashboard(self, ax):
        """
        绘制KPI仪表盘
        
        显示关键性能指标
        """
        ax.axis('off')
        
        kpis = self.statistics.get_kpis()
        
        # 创建KPI文本
        kpi_text = [
            ("总订单数", f"{kpis['total_orders']}"),
            ("完成订单", f"{kpis['completed']}"),
            ("准时率", f"{kpis['on_time_rate']:.1f}%"),
            ("平均成本/单", f"{kpis['avg_cost_per_order']:.3f}"),
            ("平均延迟", f"{kpis['avg_delay']:.2f} min"),
            ("总成本", f"{kpis['total_cost']:.1f}"),
        ]
        
        # 绘制KPI卡片
        y_positions = np.linspace(0.85, 0.15, len(kpi_text))
        
        for i, (label, value) in enumerate(kpi_text):
            y = y_positions[i]
            
            # 绘制背景框
            rect = plt.Rectangle((0.05, y - 0.08), 0.9, 0.12,
                                 fill=True, facecolor='#ecf0f1',
                                 edgecolor='#bdc3c7', linewidth=1,
                                 transform=ax.transAxes)
            ax.add_patch(rect)
            
            # 标签和数值
            ax.text(0.15, y, label, fontsize=11, fontweight='normal',
                   transform=ax.transAxes, va='center')
            ax.text(0.85, y, value, fontsize=12, fontweight='bold',
                   transform=ax.transAxes, va='center', ha='right',
                   color='#2c3e50')
        
        ax.set_title("关键性能指标 (KPIs)", fontsize=12, fontweight='bold',
                    transform=ax.transAxes, y=0.98)
    
    def plot_agent_type_distribution(self, save_path: str = None, show: bool = True):
        """
        绘制各代理类型处理订单的分布
        """
        # 从调度日志统计
        orders_by_type = defaultdict(int)
        for log in self.dispatch_log:
            orders_by_type[log['agent_type']] += len(log['orders'])
        
        if not orders_by_type:
            print("无调度数据")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        types = list(orders_by_type.keys())
        counts = [orders_by_type[t] for t in types]
        colors = ['#3498db', '#2ecc71', '#e74c3c'][:len(types)]
        
        bars = ax.bar(types, counts, color=colors, edgecolor='black')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.annotate(f'{count}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=12)
        
        ax.set_xlabel("代理类型", fontsize=11)
        ax.set_ylabel("处理订单数", fontsize=11)
        ax.set_title("各代理类型订单处理分布", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_delivery_time_histogram(self, save_path: str = None, show: bool = True):
        """
        绘制配送时间分布直方图
        """
        # 计算配送时间
        delivery_times = []
        for order in self.completed_orders:
            if order.delivery_time is not None and order.gen_time is not None:
                dt = order.delivery_time - order.gen_time
                delivery_times.append(dt)
        
        if not delivery_times:
            print("无完成订单数据")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 直方图
        n, bins, patches = ax.hist(delivery_times, bins=20, color='#3498db',
                                   edgecolor='white', alpha=0.7)
        
        # 标记30分钟截止线
        ax.axvline(x=config.HARD_DEADLINE, color='red', linestyle='--',
                  linewidth=2, label=f'截止时间 ({config.HARD_DEADLINE}min)')
        
        # 统计信息
        mean_time = np.mean(delivery_times)
        ax.axvline(x=mean_time, color='green', linestyle='-.',
                  linewidth=2, label=f'平均时间 ({mean_time:.1f}min)')
        
        ax.set_xlabel("配送时间 (分钟)", fontsize=11)
        ax.set_ylabel("订单数量", fontsize=11)
        ax.set_title("配送时间分布", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_order_cumulative(self, save_path: str = None, show: bool = True):
        """
        绘制订单累积完成曲线
        """
        status_data = self.statistics.order_status_over_time
        
        if not status_data:
            print("无订单状态数据")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        times = [d[0] for d in status_data]
        pending = [d[1] for d in status_data]
        assigned = [d[2] for d in status_data]
        completed = [d[3] for d in status_data]
        
        ax.plot(times, pending, label='待处理', color='#e74c3c', linewidth=2)
        ax.plot(times, assigned, label='配送中', color='#f39c12', linewidth=2)
        ax.plot(times, completed, label='已完成', color='#27ae60', linewidth=2)
        
        ax.set_xlabel("仿真时间 (分钟)", fontsize=11)
        ax.set_ylabel("订单数量", fontsize=11)
        ax.set_title("订单状态变化曲线", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, config.EXTENDED_END)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def generate_report(self) -> str:
        """
        生成文本分析报告
        
        Returns:
            报告字符串
        """
        kpis = self.statistics.get_kpis()
        
        report = []
        report.append("=" * 60)
        report.append("校园配送系统仿真分析报告")
        report.append("=" * 60)
        
        report.append("\n## 1. 仿真概述")
        report.append(f"- 仿真时间范围: {config.SIMULATION_START}-{config.SIMULATION_END} 分钟 (11:30-12:30)")
        report.append(f"- 扩展清理时间: {config.SIMULATION_END}-{config.EXTENDED_END} 分钟")
        report.append(f"- 时间步长: {config.TIME_STEP} 分钟")
        report.append(f"- RHC调度间隔: {config.RHC_INTERVAL} 分钟")
        
        report.append("\n## 2. 车队配置")
        for agent_type, conf in config.FLEET_CONFIG.items():
            report.append(f"- {agent_type}: 数量={conf['count']}, "
                         f"容量={conf['capacity']}, "
                         f"速度={conf['speed_multiplier']}, "
                         f"成本={conf['cost_per_order']}/单")
        
        report.append("\n## 3. 关键性能指标 (KPIs)")
        report.append(f"- 总订单数: {kpis['total_orders']}")
        report.append(f"- 完成订单: {kpis['completed']}")
        report.append(f"- 失败订单: {kpis['failed']}")
        report.append(f"- **准时率**: {kpis['on_time_rate']:.2f}%")
        report.append(f"- **平均每单成本**: {kpis['avg_cost_per_order']:.4f}")
        report.append(f"- **平均延迟**: {kpis['avg_delay']:.2f} 分钟")
        report.append(f"- 总成本: {kpis['total_cost']:.2f}")
        
        report.append("\n## 4. 成本分布")
        for agent_type, cost in kpis['cost_by_agent_type'].items():
            pct = cost / kpis['total_cost'] * 100 if kpis['total_cost'] > 0 else 0
            report.append(f"- {agent_type}: {cost:.2f} ({pct:.1f}%)")
        
        report.append("\n## 5. 调度统计")
        dispatch_summary = self._get_dispatch_summary()
        report.append(f"- 总调度次数: {dispatch_summary['total_dispatches']}")
        for agent_type, count in dispatch_summary['dispatches_by_type'].items():
            report.append(f"- {agent_type} 调度次数: {count}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def _get_dispatch_summary(self) -> Dict:
        """获取调度统计摘要"""
        summary = {
            "total_dispatches": len(self.dispatch_log),
            "dispatches_by_type": defaultdict(int),
            "orders_by_type": defaultdict(int),
        }
        
        for log in self.dispatch_log:
            agent_type = log['agent_type']
            summary['dispatches_by_type'][agent_type] += 1
            summary['orders_by_type'][agent_type] += len(log['orders'])
        
        return summary


def print_kpi_table(statistics: SimulationStatistics):
    """
    打印KPI表格
    
    Args:
        statistics: 仿真统计对象
    """
    kpis = statistics.get_kpis()
    
    print("\n" + "=" * 50)
    print(f"{'指标':<20} | {'值':>15}")
    print("=" * 50)
    print(f"{'总订单数':<20} | {kpis['total_orders']:>15}")
    print(f"{'完成订单':<20} | {kpis['completed']:>15}")
    print(f"{'失败订单':<20} | {kpis['failed']:>15}")
    print(f"{'准时送达':<20} | {kpis['on_time_deliveries']:>15}")
    print(f"{'准时率':<20} | {kpis['on_time_rate']:>14.2f}%")
    print(f"{'平均成本/单':<20} | {kpis['avg_cost_per_order']:>15.4f}")
    print(f"{'平均延迟(分钟)':<20} | {kpis['avg_delay']:>15.2f}")
    print(f"{'总成本':<20} | {kpis['total_cost']:>15.2f}")
    print("=" * 50)


if __name__ == "__main__":
    # 测试分析模块
    from simulator import Environment
    import numpy as np
    
    np.random.seed(config.RANDOM_SEED)
    
    print("运行仿真...")
    env = Environment(verbose=False)
    results = env.run()
    
    print("\n生成分析报告...")
    analyzer = SimulationAnalyzer(results)
    
    # 打印报告
    report = analyzer.generate_report()
    print(report)
    
    # 生成图表
    analyzer.generate_all_plots(show=True)

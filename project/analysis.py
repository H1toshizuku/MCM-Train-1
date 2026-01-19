"""
analysis.py - Visualization and KPI Analysis Module

This module implements visualization and KPI calculation functions:
- Demand Heatmap
- Fleet Usage Stacked Area Chart
- Order Status Pie Chart
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


# Set font for English labels
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


class SimulationAnalyzer:
    """Simulation Result Analyzer"""
    
    def __init__(self, results: Dict):
        """
        Initialize analyzer
        
        Args:
            results: Simulation results dictionary
        """
        self.all_orders = results.get("all_orders", [])
        self.completed_orders = results.get("completed_orders", [])
        self.failed_orders = results.get("failed_orders", [])
        self.statistics: SimulationStatistics = results.get("statistics")
        self.dispatch_log = results.get("dispatch_log", [])
    
    def generate_all_plots(self, save_path: str = None, show: bool = True):
        """
        Generate all visualization charts
        
        Args:
            save_path: Save path (if specified, save the image)
            show: Whether to display the chart
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Campus Delivery System Simulation Report", fontsize=16, fontweight='bold')
        
        # 1. Demand Heatmap
        self._plot_demand_heatmap(axes[0, 0])
        
        # 2. Fleet Usage
        self._plot_fleet_usage(axes[0, 1])
        
        # 3. Order Status Pie
        self._plot_order_status_pie(axes[1, 0])
        
        # 4. KPI Dashboard
        self._plot_kpi_dashboard(axes[1, 1])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _plot_demand_heatmap(self, ax):
        """
        Plot demand heatmap (bar chart)
        
        Shows order count per 10-minute interval
        """
        # Count orders by time interval
        interval_counts = defaultdict(int)
        for order in self.all_orders:
            interval_idx = int(order.gen_time // 10) * 10
            interval_counts[interval_idx] += 1
        
        # Prepare data
        intervals = list(range(0, 60, 10))
        counts = [interval_counts.get(i, 0) for i in intervals]
        labels = [f"{i}-{i+10}" for i in intervals]
        
        # Color mapping
        colors = plt.cm.YlOrRd(np.array(counts) / max(counts))
        
        bars = ax.bar(labels, counts, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.annotate(f'{count}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel("Time Interval (minutes)", fontsize=11)
        ax.set_ylabel("Order Count", fontsize=11)
        ax.set_title("Order Demand Distribution", fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(counts) * 1.15)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_fleet_usage(self, ax):
        """
        Plot fleet usage stacked area chart
        
        Shows active agent count over time
        """
        usage_data = self.statistics.agent_usage_over_time
        
        if not usage_data:
            ax.text(0.5, 0.5, "No agent usage data", ha='center', va='center')
            ax.set_title("Fleet Usage", fontsize=12, fontweight='bold')
            return
        
        # Prepare data
        times = [d[0] for d in usage_data]
        
        human_usage = [d[1].get(config.AGENT_TYPE_HUMAN, 0) for d in usage_data]
        agv_usage = [d[1].get(config.AGENT_TYPE_AGV, 0) for d in usage_data]
        drone_usage = [d[1].get(config.AGENT_TYPE_DRONE, 0) for d in usage_data]
        
        # Stacked area chart
        ax.stackplot(times, human_usage, agv_usage, drone_usage,
                    labels=['Human Rider', 'AGV', 'Drone'],
                    colors=['#3498db', '#2ecc71', '#e74c3c'],
                    alpha=0.8)
        
        ax.set_xlabel("Simulation Time (minutes)", fontsize=11)
        ax.set_ylabel("Active Agent Count", fontsize=11)
        ax.set_title("Fleet Usage Over Time", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_xlim(0, config.EXTENDED_END)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_order_status_pie(self, ax):
        """
        Plot order status pie chart
        
        Shows on-time vs late vs failed ratio
        """
        kpis = self.statistics.get_kpis()
        
        # Data
        on_time = kpis['on_time_deliveries']
        late = kpis['late_deliveries']
        failed = kpis['failed']
        
        # Filter zero values
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
        if late > 0 and failed == 0:
            labels.append('Late')
            sizes.append(late)
            colors_list.append(color_map['Late'])
        if failed > 0:
            labels.append('Failed')
            sizes.append(failed)
            colors_list.append(color_map['Failed'])
        
        if not sizes:
            ax.text(0.5, 0.5, "No order data", ha='center', va='center')
            ax.set_title("Order Status Distribution", fontsize=12, fontweight='bold')
            return
        
        # Pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors_list,
            autopct='%1.1f%%', startangle=90,
            explode=[0.02] * len(sizes),
            shadow=True
        )
        
        # Set font
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        ax.set_title("Order Status Distribution", fontsize=12, fontweight='bold')
    
    def _plot_kpi_dashboard(self, ax):
        """
        Plot KPI dashboard
        
        Shows key performance indicators
        """
        ax.axis('off')
        
        kpis = self.statistics.get_kpis()
        
        # Create KPI text
        kpi_text = [
            ("Total Orders", f"{kpis['total_orders']}"),
            ("Completed", f"{kpis['completed']}"),
            ("On-time Rate", f"{kpis['on_time_rate']:.1f}%"),
            ("Avg Cost/Order", f"{kpis['avg_cost_per_order']:.3f}"),
            ("Avg Delay", f"{kpis['avg_delay']:.2f} min"),
            ("Total Cost", f"{kpis['total_cost']:.1f}"),
        ]
        
        # Draw KPI cards
        y_positions = np.linspace(0.85, 0.15, len(kpi_text))
        
        for i, (label, value) in enumerate(kpi_text):
            y = y_positions[i]
            
            # Draw background box
            rect = plt.Rectangle((0.05, y - 0.08), 0.9, 0.12,
                                 fill=True, facecolor='#ecf0f1',
                                 edgecolor='#bdc3c7', linewidth=1,
                                 transform=ax.transAxes)
            ax.add_patch(rect)
            
            # Label and value
            ax.text(0.15, y, label, fontsize=11, fontweight='normal',
                   transform=ax.transAxes, va='center')
            ax.text(0.85, y, value, fontsize=12, fontweight='bold',
                   transform=ax.transAxes, va='center', ha='right',
                   color='#2c3e50')
        
        ax.set_title("Key Performance Indicators (KPIs)", fontsize=12, fontweight='bold',
                    transform=ax.transAxes, y=0.98)
    
    def plot_agent_type_distribution(self, save_path: str = None, show: bool = True):
        """
        Plot order distribution by agent type
        """
        # Count from dispatch log
        orders_by_type = defaultdict(int)
        for log in self.dispatch_log:
            orders_by_type[log['agent_type']] += len(log['orders'])
        
        if not orders_by_type:
            print("No dispatch data")
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
        
        ax.set_xlabel("Agent Type", fontsize=11)
        ax.set_ylabel("Orders Delivered", fontsize=11)
        ax.set_title("Order Distribution by Agent Type", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_delivery_time_histogram(self, save_path: str = None, show: bool = True):
        """
        Plot delivery time distribution histogram
        """
        # Calculate delivery times
        delivery_times = []
        for order in self.completed_orders:
            if order.delivery_time is not None and order.gen_time is not None:
                dt = order.delivery_time - order.gen_time
                delivery_times.append(dt)
        
        if not delivery_times:
            print("No completed order data")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        n, bins, patches = ax.hist(delivery_times, bins=20, color='#3498db',
                                   edgecolor='white', alpha=0.7)
        
        # Mark 30-minute deadline
        ax.axvline(x=config.HARD_DEADLINE, color='red', linestyle='--',
                  linewidth=2, label=f'Deadline ({config.HARD_DEADLINE}min)')
        
        # Statistics
        mean_time = np.mean(delivery_times)
        ax.axvline(x=mean_time, color='green', linestyle='-.',
                  linewidth=2, label=f'Mean Time ({mean_time:.1f}min)')
        
        ax.set_xlabel("Delivery Time (minutes)", fontsize=11)
        ax.set_ylabel("Order Count", fontsize=11)
        ax.set_title("Delivery Time Distribution", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_order_cumulative(self, save_path: str = None, show: bool = True):
        """
        Plot order status over time
        """
        status_data = self.statistics.order_status_over_time
        
        if not status_data:
            print("No order status data")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        times = [d[0] for d in status_data]
        pending = [d[1] for d in status_data]
        assigned = [d[2] for d in status_data]
        completed = [d[3] for d in status_data]
        
        ax.plot(times, pending, label='Pending', color='#e74c3c', linewidth=2)
        ax.plot(times, assigned, label='Delivering', color='#f39c12', linewidth=2)
        ax.plot(times, completed, label='Completed', color='#27ae60', linewidth=2)
        
        ax.set_xlabel("Simulation Time (minutes)", fontsize=11)
        ax.set_ylabel("Order Count", fontsize=11)
        ax.set_title("Order Status Over Time", fontsize=14, fontweight='bold')
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
        Generate text analysis report
        
        Returns:
            Report string
        """
        kpis = self.statistics.get_kpis()
        
        report = []
        report.append("=" * 60)
        report.append("Campus Delivery System Simulation Report")
        report.append("=" * 60)
        
        report.append("\n## 1. Simulation Overview")
        report.append(f"- Simulation Time Range: {config.SIMULATION_START}-{config.SIMULATION_END} min (11:30-12:30)")
        report.append(f"- Extended Cleanup Time: {config.SIMULATION_END}-{config.EXTENDED_END} min")
        report.append(f"- Time Step: {config.TIME_STEP} min")
        report.append(f"- RHC Dispatch Interval: {config.RHC_INTERVAL} min")
        
        report.append("\n## 2. Fleet Configuration")
        for agent_type, conf in config.FLEET_CONFIG.items():
            report.append(f"- {agent_type}: count={conf['count']}, "
                         f"capacity={conf['capacity']}, "
                         f"speed={conf['speed_multiplier']}, "
                         f"cost={conf['cost_per_order']}/order")
        
        report.append("\n## 3. Key Performance Indicators (KPIs)")
        report.append(f"- Total Orders: {kpis['total_orders']}")
        report.append(f"- Completed Orders: {kpis['completed']}")
        report.append(f"- Failed Orders: {kpis['failed']}")
        report.append(f"- **On-time Rate**: {kpis['on_time_rate']:.2f}%")
        report.append(f"- **Avg Cost per Order**: {kpis['avg_cost_per_order']:.4f}")
        report.append(f"- **Avg Delay**: {kpis['avg_delay']:.2f} min")
        report.append(f"- Total Cost: {kpis['total_cost']:.2f}")
        
        report.append("\n## 4. Cost Distribution")
        for agent_type, cost in kpis['cost_by_agent_type'].items():
            pct = cost / kpis['total_cost'] * 100 if kpis['total_cost'] > 0 else 0
            report.append(f"- {agent_type}: {cost:.2f} ({pct:.1f}%)")
        
        report.append("\n## 5. Dispatch Statistics")
        dispatch_summary = self._get_dispatch_summary()
        report.append(f"- Total Dispatches: {dispatch_summary['total_dispatches']}")
        for agent_type, count in dispatch_summary['dispatches_by_type'].items():
            report.append(f"- {agent_type} Dispatches: {count}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def _get_dispatch_summary(self) -> Dict:
        """Get dispatch statistics summary"""
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
    Print KPI table
    
    Args:
        statistics: Simulation statistics object
    """
    kpis = statistics.get_kpis()
    
    print("\n" + "=" * 50)
    print(f"{'Metric':<20} | {'Value':>15}")
    print("=" * 50)
    print(f"{'Total Orders':<20} | {kpis['total_orders']:>15}")
    print(f"{'Completed':<20} | {kpis['completed']:>15}")
    print(f"{'Failed':<20} | {kpis['failed']:>15}")
    print(f"{'On-time Deliveries':<20} | {kpis['on_time_deliveries']:>15}")
    print(f"{'On-time Rate':<20} | {kpis['on_time_rate']:>14.2f}%")
    print(f"{'Avg Cost/Order':<20} | {kpis['avg_cost_per_order']:>15.4f}")
    print(f"{'Avg Delay (min)':<20} | {kpis['avg_delay']:>15.2f}")
    print(f"{'Total Cost':<20} | {kpis['total_cost']:>15.2f}")
    print("=" * 50)


if __name__ == "__main__":
    # Test analysis module
    from simulator import Environment
    import numpy as np
    
    np.random.seed(config.RANDOM_SEED)
    
    print("Running simulation...")
    env = Environment(verbose=False)
    results = env.run()
    
    print("\nGenerating analysis report...")
    analyzer = SimulationAnalyzer(results)
    
    # Print report
    report = analyzer.generate_report()
    print(report)
    
    # Generate charts
    analyzer.generate_all_plots(show=True)

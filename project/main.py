"""
main.py - 仿真系统入口

This is the main entry point for the Heterogeneous Campus Delivery System simulation.

Mathematical Modeling Contest - Problem B: Campus Delivery System Design

Usage:
    python main.py [--verbose] [--save-plots] [--output-dir <path>]
"""

import argparse
import os
import sys
import numpy as np
from datetime import datetime

# 导入项目模块
import config
from generator import DemandGenerator, generate_demand_statistics, print_demand_statistics
from simulator import Environment
from analysis import SimulationAnalyzer, print_kpi_table


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="校园配送系统离散事件仿真"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="保存图表到文件"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="输出目录路径"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不显示图表（适用于无GUI环境）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="随机数种子"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 打印标题
    print("=" * 70)
    print("  校园配送系统离散事件仿真 (Heterogeneous Campus Delivery System)")
    print("  Mathematical Modeling Contest - Problem B")
    print("=" * 70)
    print(f"\n运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"随机种子: {args.seed}")
    
    # ========================================================================
    # Step 1: 生成需求
    # ========================================================================
    print("\n" + "-" * 50)
    print("Step 1: 生成需求 (NHPP Demand Generation)")
    print("-" * 50)
    
    generator = DemandGenerator(seed=args.seed)
    orders = generator.generate_orders()
    
    stats = generate_demand_statistics(orders)
    print_demand_statistics(stats)
    
    # ========================================================================
    # Step 2: 运行仿真
    # ========================================================================
    print("\n" + "-" * 50)
    print("Step 2: 运行仿真 (Rolling Horizon Control)")
    print("-" * 50)
    
    env = Environment(orders=orders, verbose=args.verbose)
    results = env.run()
    
    # ========================================================================
    # Step 3: 分析结果
    # ========================================================================
    print("\n" + "-" * 50)
    print("Step 3: 分析结果 (KPI Analysis)")
    print("-" * 50)
    
    # 打印KPI表格
    print_kpi_table(results["statistics"])
    
    # 创建分析器
    analyzer = SimulationAnalyzer(results)
    
    # 打印详细报告
    report = analyzer.generate_report()
    print(report)
    
    # ========================================================================
    # Step 4: 生成可视化
    # ========================================================================
    print("\n" + "-" * 50)
    print("Step 4: 生成可视化 (Visualization)")
    print("-" * 50)
    
    # 创建输出目录
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"输出目录: {args.output_dir}")
    
    # 生成综合图表
    show_plots = not args.no_show
    save_path = os.path.join(args.output_dir, "simulation_report.png") if args.save_plots else None
    
    try:
        analyzer.generate_all_plots(save_path=save_path, show=show_plots)
    except Exception as e:
        print(f"图表生成警告: {e}")
        print("（可能是无GUI环境，尝试保存图表）")
        
        if args.save_plots:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            analyzer.generate_all_plots(save_path=save_path, show=False)
    
    # 生成额外图表
    if args.save_plots:
        try:
            analyzer.plot_agent_type_distribution(
                save_path=os.path.join(args.output_dir, "agent_distribution.png"),
                show=False
            )
            analyzer.plot_delivery_time_histogram(
                save_path=os.path.join(args.output_dir, "delivery_time_hist.png"),
                show=False
            )
            analyzer.plot_order_cumulative(
                save_path=os.path.join(args.output_dir, "order_cumulative.png"),
                show=False
            )
            print("额外图表已保存")
        except Exception as e:
            print(f"额外图表生成警告: {e}")
    
    # ========================================================================
    # 完成
    # ========================================================================
    print("\n" + "=" * 70)
    print("仿真完成!")
    print("=" * 70)
    
    # 返回结果供进一步分析
    return results


def run_sensitivity_analysis():
    """
    敏感性分析
    
    测试不同参数配置下的系统性能
    """
    print("\n" + "=" * 70)
    print("敏感性分析 (Sensitivity Analysis)")
    print("=" * 70)
    
    results_list = []
    
    # 测试不同的RHC间隔
    rhc_intervals = [3, 5, 10]
    
    for interval in rhc_intervals:
        print(f"\n测试 RHC_INTERVAL = {interval} 分钟...")
        
        # 临时修改配置
        original_interval = config.RHC_INTERVAL
        config.RHC_INTERVAL = interval
        
        # 重置随机种子
        np.random.seed(config.RANDOM_SEED)
        
        # 生成订单和运行仿真
        generator = DemandGenerator()
        orders = generator.generate_orders()
        
        env = Environment(orders=orders, verbose=False)
        results = env.run()
        
        kpis = results["statistics"].get_kpis()
        results_list.append({
            "rhc_interval": interval,
            "on_time_rate": kpis["on_time_rate"],
            "avg_cost": kpis["avg_cost_per_order"],
            "avg_delay": kpis["avg_delay"],
        })
        
        # 恢复配置
        config.RHC_INTERVAL = original_interval
    
    # 打印结果
    print("\n敏感性分析结果:")
    print("-" * 60)
    print(f"{'RHC间隔':>10} | {'准时率':>10} | {'平均成本':>10} | {'平均延迟':>10}")
    print("-" * 60)
    for r in results_list:
        print(f"{r['rhc_interval']:>10} | {r['on_time_rate']:>9.2f}% | "
              f"{r['avg_cost']:>10.4f} | {r['avg_delay']:>10.2f}")
    print("-" * 60)
    
    return results_list


def run_monte_carlo_simulation(n_runs: int = 10):
    """
    蒙特卡洛仿真
    
    多次运行仿真以评估系统的统计特性
    
    Args:
        n_runs: 运行次数
    """
    print("\n" + "=" * 70)
    print(f"蒙特卡洛仿真 (Monte Carlo Simulation, n={n_runs})")
    print("=" * 70)
    
    all_kpis = []
    
    for i in range(n_runs):
        seed = config.RANDOM_SEED + i
        np.random.seed(seed)
        
        generator = DemandGenerator(seed=seed)
        orders = generator.generate_orders()
        
        env = Environment(orders=orders, verbose=False)
        results = env.run()
        
        kpis = results["statistics"].get_kpis()
        all_kpis.append(kpis)
        
        print(f"  运行 {i+1}/{n_runs}: 准时率={kpis['on_time_rate']:.2f}%")
    
    # 统计分析
    on_time_rates = [k["on_time_rate"] for k in all_kpis]
    avg_costs = [k["avg_cost_per_order"] for k in all_kpis]
    avg_delays = [k["avg_delay"] for k in all_kpis]
    
    print("\n蒙特卡洛仿真结果统计:")
    print("-" * 50)
    print(f"{'指标':<15} | {'均值':>10} | {'标准差':>10} | {'范围':>20}")
    print("-" * 50)
    print(f"{'准时率 (%)':<15} | {np.mean(on_time_rates):>10.2f} | "
          f"{np.std(on_time_rates):>10.2f} | "
          f"{min(on_time_rates):.2f} - {max(on_time_rates):.2f}")
    print(f"{'平均成本':<15} | {np.mean(avg_costs):>10.4f} | "
          f"{np.std(avg_costs):>10.4f} | "
          f"{min(avg_costs):.4f} - {max(avg_costs):.4f}")
    print(f"{'平均延迟 (min)':<15} | {np.mean(avg_delays):>10.2f} | "
          f"{np.std(avg_delays):>10.2f} | "
          f"{min(avg_delays):.2f} - {max(avg_delays):.2f}")
    print("-" * 50)
    
    return all_kpis


if __name__ == "__main__":
    # 运行主仿真
    results = main()
    
    # 可选：运行敏感性分析
    # run_sensitivity_analysis()
    
    # 可选：运行蒙特卡洛仿真
    # run_monte_carlo_simulation(n_runs=10)

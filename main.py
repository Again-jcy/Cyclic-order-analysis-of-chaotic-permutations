import numpy as np
from evaluate import Evaluate
from generate import Generate
import matplotlib.pyplot as plt

def main():
    # 设置不同混沌映射的参数
    params = {
        'logistic': (Generate.logistic_map, 3.8, None),
        'tent': (Generate.tent_map, 1.9, None),
        'icmic': (Generate.icmic_map, np.pi, None),
        'logistic_tent': (Generate.logistic_tent_map, 3.9, None),
        'henon': (Generate.henon_map, 1.4, 0.3)
    }

    chosen_map = 'logistic'  # 选择混沌映射类型
    
    if chosen_map not in params:
        raise ValueError(f"Unsupported map: {chosen_map}")

    map_func, param1, param2 = params[chosen_map] # 传递所选混沌映射的参数
    
    M = 1000
    N = 100
    seeds = [(np.random.rand(), np.random.rand()) for _ in range(10000)] # 生成10000个随机种子
    N_values = list(range(20, 1001, 20)) # 绘图时N的取值范围

    evaluator = Evaluate(map_func, param1, param2)
    permutation_list=[]
    total_order_list=[]
    sensitivity_list=[]
    
    # 生成置乱表并进行测评
    for x0, y0 in seeds:
        permutation = evaluator.generator.generate_permutation(x0, y0, M, N)  
        permutation_list.append(permutation)
        # 统计置乱表的循环情况
        cycle_length_counts, total_order = evaluator.find_cycles(permutation)
        total_order_list.append(total_order)
        print(f"循环圈的长度和个数: {dict(cycle_length_counts)}")
        print(f"置乱表的总阶: {total_order}")
        # 敏感性分析
        sensitivity = evaluator.sensitivity_tests(permutation, x0, y0, M, N)
        print(f"修改种子前后置乱表的相似程度: {sensitivity:.5f}")
        sensitivity_list.append(sensitivity)
 
    average_order = np.mean(total_order_list)
    print(f"平均的阶:{average_order}")
    average_sensitivity = np.mean(sensitivity_list) #
    print(f"修改种子前后置乱表的平均相似程度 : {average_sensitivity:.5f}")
    p_values, combined_p_value = evaluator.randomness_tests(permutation_list)
    #print(p_values)
    print(f"p值:{combined_p_value}")
   
    # 绘制平均阶-N曲线
    #average_orders = evaluator.plot_average_order_vs_N(M, N_values, seeds)
    # 显示图表
    #plt.show()

if __name__ == "__main__":
    main()

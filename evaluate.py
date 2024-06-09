import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import combine_pvalues
from scipy.stats import chisquare
from generate import Generate

class Evaluate:
    def __init__(self, map_func, *params):
        self.generator = Generate(map_func, *params)
    
    def find_cycles(self, permutation):
        """
        这个函数用于评测给定置乱表的循环情况。
        包括：置乱表中循环圈长度的种数，每种长度的循环圈个数以及总的循环长度（阶）。
        """
        n = len(permutation)
        visited = [False] * n # 用于标记这个元素是否已经被访问过
        cycles = []
        
        # 统计置乱表中的每个循环
        for i in range(n):
            if not visited[i]:
                cycle = []
                current = i
                while not visited[current]:
                    visited[current] = True
                    cycle.append(current)
                    current = permutation[current]
                cycles.append(cycle)
        
        cycle_lengths = [len(cycle) for cycle in cycles] # 计算每个循环的长度
        total_order = np.lcm.reduce(cycle_lengths) # 计算所有循环长度的最小公倍数，得到总的循环长度（阶）
        
        cycle_length_counts = defaultdict(int) # 创建一个字典，用于存储每种长度的循环圈的数量
        for length in cycle_lengths:
            cycle_length_counts[length] += 1 # 统计每种长度的循环圈的数量
        
        return cycle_length_counts, total_order
  
    def randomness_tests(self, permutations):
        """
        这个函数用于测评置乱表的随机性，即每个元素在置乱表中是否是均匀分布的。
        """
        N = len(permutations[0])
        position_counts = np.zeros((N, N)) # 初始化位置计数矩阵，第(i,j)个元素表示i在第j个位置出现的次数
        
        # 统计每个元素在各个位置上的出现次数
        for permutation in permutations:
            for i, pos in enumerate(permutation):
                position_counts[i, pos] += 1
        
        # 每个元素在各个位置上的期望出现频率都应该是1/N，总的期望出现次数就是len(permutations) / N
        expected_counts = np.full(N, len(permutations) / N)
        p_values = []
        
        # 进行卡方检验计算每个位置的p值
        for counts in position_counts:
            p_value = chisquare(counts, f_exp=expected_counts).pvalue
            p_values.append(p_value)
        
        # 合并多个p值以获得一个总体p值
        combined_p_value = combine_pvalues(p_values)[1]
        
        return p_values, combined_p_value


    def sensitivity_tests(self, permutation, x0, y0, M, N, delta=1e-5):
        """
        这个函数用于敏感性测试，测评置乱表对初始种子的敏感度。
        """
        perm_original = permutation
        x0_new, y0_new = x0 + delta, y0 + delta # 将初始种子改变一个微小的量
        perm_modified = self.generator.generate_permutation(x0_new, y0_new, M, N)
        similarity = np.sum(np.array(perm_original) == np.array(perm_modified)) / N # 计算改变种子前后两置乱表的相似度
        
        return similarity
    
    def evaluate_permutations(self, M, N, seeds):
        """
        这个函数用于计算不同种子生成的置乱表的平均阶。
        这里定义这个函数的目的是方便后面绘制“平均阶-N”曲线,主函数里其实用不上。
        """
        total_orders = []
        for x0, y0 in seeds:
            permutation = self.generator.generate_permutation(x0, y0, M, N)
            _, total_order = self.find_cycles(permutation)
            total_orders.append(total_order)
        
        average_order = np.mean(total_orders) # 计算平均阶
        return average_order, total_orders

    def plot_average_order_vs_N(self, M, N_values, seeds):
        """
        这个函数用于绘制“平均阶-N”的曲线。
        """
        average_orders = []
        for N in N_values:
            avg_order, _ = self.evaluate_permutations(M, N, seeds) # 计算每个N值对应的平均阶
            average_orders.append(avg_order)

        # 作出“平均阶-N”的曲线
        plt.plot(N_values, average_orders, marker='o')
        plt.xlabel('N')
        plt.ylabel('Average Order')
        plt.title('Average Order vs N')
        plt.grid(True)
        
        return average_orders

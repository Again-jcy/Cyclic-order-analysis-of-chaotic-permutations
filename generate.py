import numpy as np

class Generate:
    def __init__(self, map_func, *params):
        self.map_func = map_func
        self.params = params

    @staticmethod
    def logistic_map(x, y, r,none,iterations):
        for _ in range(iterations):
            x = r * x * (1 - x)
        return x, y

    @staticmethod
    def tent_map(x, y, r,none, iterations):
        for _ in range(iterations):
            if x < 0.5:
                x = r * x
            else:
                x = r * (1 - x)
        return x, y

    @staticmethod
    def icmic_map(x,y,r,none,iterations):
        for _ in range(iterations):
            x = np.sin(r * x)
        return x,y

    @staticmethod
    def logistic_tent_map(x, y, r, none,iterations):
        for _ in range(iterations):
            if x < 0.5:
                x = (r * x * (1 - x) + (4 - r) * x / 2) % 1
            else:
                x = (r * x * (1 - x) + (4 - r) * (1 - x) / 2) % 1
        return x, y

    @staticmethod
    def henon_map(x, y, a, b, iterations):
        for _ in range(iterations):
            x, y = 1 - a * x**2 + y, b * x
            # 防止数据溢出
            if abs(x) > 1e10 or abs(y) > 1e10:
                x, y = np.random.rand(), np.random.rand()
        return x, y

    def generate_permutation(self, x0, y0, M, N):
        # 初始化混沌映射
        x, y = self.map_func(x0, y0, *self.params, M)
        
        # 生成N个值
        values = []
        for _ in range(N):
            x, y = self.map_func(x, y, *self.params, 1)
            values.append(x)  
        
        # 对这N个值进行排序，得到置乱表
        sorted_indices = np.argsort(values)
        permutation = np.argsort(sorted_indices)
        
        return permutation

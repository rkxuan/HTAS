class UnionFind:
    def __init__(self):
        self.parent = {}  
        self.rank = {}   
        self.components = {}  
    
    def contain(self, x):
        return x in self.parent

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.components[x] = {x} 
            return x
        
        # 路径压缩
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  
        
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        
        self.components[root_x].update(self.components[root_y])
        del self.components[root_y] 
        
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
    
    def get_component(self, x):
        root = self.find(x)
        return list(self.components[root])  
    
    def get_all_components(self):
        return {root: nodes.copy() for root, nodes in self.components.items()}
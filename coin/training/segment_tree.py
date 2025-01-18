import operator

class SegmentTree:
    def __init__(self, capacity, operation, neutral_element):
        """
        세그먼트 트리 초기화
        capacity: 트리의 용량
        operation: 트리에서 수행할 연산 (min이나 sum)
        neutral_element: 연산의 항등원
        """
        self.capacity = capacity
        self.operation = operation
        self.neutral_element = neutral_element
        
        # 트리의 높이 계산
        self.tree_height = 1
        while 2**self.tree_height < capacity:
            self.tree_height += 1
            
        # 트리 배열 초기화
        self.tree_size = 2**(self.tree_height + 1)
        self.tree = [self.neutral_element for _ in range(self.tree_size)]
        
        self.values = [self.neutral_element for _ in range(capacity)]
        
    def _operate_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self.tree[node]
        
        mid = (node_start + node_end) // 2
        
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )
    
    def operate(self, start=0, end=None):
        if end is None:
            end = self.capacity
        if end < 0:
            end += self.capacity
            
        end -= 1
        
        return self._operate_helper(start, end, 1, 0, self.capacity - 1)
    
    def __setitem__(self, idx, val):
        # 값 배열 업데이트
        self.values[idx] = val
        
        # 트리 노드 업데이트
        idx_tree = 2**self.tree_height + idx
        self.tree[idx_tree] = val
        idx_tree //= 2
        
        while idx_tree >= 1:
            self.tree[idx_tree] = self.operation(
                self.tree[2 * idx_tree],
                self.tree[2 * idx_tree + 1]
            )
            idx_tree //= 2
    
    def __getitem__(self, idx):
        return self.values[idx]

class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity=capacity, operation=operator.add, neutral_element=0.0)
    
    def sum(self, start=0, end=None):
        return super().operate(start, end)
    
    def find_prefixsum_idx(self, prefixsum):
        if prefixsum < 0 or prefixsum > self.sum():
            raise ValueError(f"Prefix sum {prefixsum} is out of bounds.")
            
        idx = 1
        while idx < 2**self.tree_height:
            if self.tree[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self.tree[2 * idx]
                idx = 2 * idx + 1
                
        return idx - 2**self.tree_height

class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity=capacity, operation=min, neutral_element=float('inf'))
    
    def min(self, start=0, end=None):
        return super().operate(start, end) 
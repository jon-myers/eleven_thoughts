import numpy as np
import random

def get_partition(n):
    """Gets the partition"""
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
            x = a[(k - 1)] + 1
            k -= 1
            while 2 * x <= y:
                    a[k] = x
                    y -= x
                    k += 1
            l = k + 1
            while x <= y:
                    a[k] = x
                    a[l] = y
                    yield a[:k + 2]
                    x += 1
                    y -= 1

            a[k] = x + y
            y = x + y - 1
            yield a[:k + 1]

def select_partition(n):
    p = list(get_partition(n))
    choice = np.random.randint(0, len(p))
    return p[choice]

#
# # select_partition(10)
# # p = list(get_partition(10))
# # print(p)
#
# def get_anti_partition(group_array):
#     partition = get_partition(6)
# class Layer:
#     def __init__(self, prev_layer):
#         self.floor_plan = [select_partition(sum(i)) for i in prev_layer]
#
# z = Layer([[1, 1, 1, 1]])
#
# print(z.floor_plan)
#
# class Node:
#     def __init__(self, num_of_strands, layer_index):
#         self.num_of_strands = num_of_strands
#         self.layer_index = layer_index
#         self.split()
#
#     def split(self):
#         self.partition = select_partition(self.num_of_strands)
#
#
# tree = Node(10,0)
# print(tree.partition)

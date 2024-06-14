# import torch
# import numpy as np
# left_link_ori = [(1,6),(2,6),(3,6),(4,2),(5,3),(7,6),(8,6),(9,7),(10,8)]
# left_link = [(i-1,j-1) for (i,j) in left_link_ori]
# back = [(j,i) for (i,j) in left_link]
# A = np.zeros((10,10))
# B = np.zeros((10,10))
# for i ,j in left_link:
#     A[j,i] = 1
#
# print(A)
#
# for i ,j in back:
#     B[j,i] = 1
#
# print(B)

import numpy as np
a1 = np.random.rand(10,10)
a2 = np.random.rand(10,10)
A = np.stack((a1,a2))
print(len(A))
b = np.random.rand(25,10)
c = b @ A[0]
print(c.shape)
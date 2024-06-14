import numpy as np
# 用于生成稀疏链接矩阵
def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))  # 首先创建一个大小为 num_in num_out的全零矩阵A
    # 使用 for 循环遍历 link 列表中的每个元素 (i, j)，将 A 的第 i 行、第 j 列的元素设置为 1，表示节点 i 和节点 j 之间存在连接。
    for i, j in link:
        A[i, j] = 1
    # 接下来，通过计算 A 的列和，使用 np.sum(A, axis=0, keepdims=True)，生成一个与 A 列数相同的行向量，保持维度一致。
    # 最后，代码将 A 中的每个元素除以对应列和，得到归一化后的矩阵 A_norm。
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm

# 用于将边的列表转化为邻接矩阵 检查link列表中每个元素(i,j)，将全零矩阵A里的对应值设为1
def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

# 用于生成k倍的尺度图
def get_k_scale_graph(scale, A):
    # 若给定的尺度是1 直接返回输入的邻接矩阵A
    if scale == 1:
        return A
    # 创建一个与A大小相同的全零矩阵An，用来储存生成的k倍尺度的图
    An = np.zeros_like(A)
    # np.eye()创建一个单位矩阵
    A_power = np.eye(A.shape[0])
    # 利用矩阵乘法操作 A_power = A_power @ A，将邻接矩阵 A 迭代相乘 k 次，并将结果累加到 An 中。
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    #  An 中大于 0 的元素设置为 1，以确保生成的图是二值的
    An[An > 0] = 1
    return An

def normalize_digraph(A):
    # 计算每个节点的出度之和
    Dl = np.sum(A, 0)
    h, w = A.shape
    # 创建一个全零矩阵 用于储存归一化因子
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            # 计算归一化因子
            Dn[i, i] = Dl[i] ** (-1)
    # 计算归一化后的邻接矩阵
    AD = np.dot(A, Dn)
    return AD

# 每个节点属于身体哪个部位
def get_R(num_node,joint_part_body_link):
    # 首先创建一个大小为(25,num_node)的全零矩阵
    R = np.zeros((25,num_node))
    # 使用嵌套的 for 循环遍历 joint_part_body_link 列表中的每个元素。
    # joint_part_body_link 是一个二维列表，每个子列表表示一个身体部位，包含了该部位涉及的节点索引。
    # 在循环中，将 R 中对应节点索引和身体部位的位置设置为 1，表示该节点属于该身体部位。
    for i in range(len(joint_part_body_link)):
        for j in range(len(joint_part_body_link[i])):
            R[joint_part_body_link[i][j],i] = 1
    # print("R:------------")
    # print(R)
    return R

def get_left(num_node,self_link, inward, outward,part_item):
    # 首先创建一个空列表 A_cross_level，用于存储计算得到的连接矩阵
    A_cross_level = []
    # I = edge2mat(self_link,num_node)
    In = normalize_digraph(edge2mat(inward,num_node))
    Out = normalize_digraph(edge2mat(outward,num_node))
    A = np.stack((In,Out)) # 获得身体部位的自身连接，以及每个身体部位的相邻连接
    R = get_R(num_node,part_item) # 获得每个节点是属于哪个身体部位的
    for i in range(len(A)):
        A_cross_level.append(R @ A[i] @ np.transpose(R,axes=[1,0]))
    # print("A_cross_level:------------")
    # print(A_cross_level)
    # 返回cross_level[1]是指左侧连接的矩阵   cross_level[0]是指自连接的矩阵
    return A_cross_level[1]


# 用于生成空间图
def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)   # 调用edgemat函数将连接列表self_link转化为 邻接矩阵
    In = normalize_digraph(edge2mat(inward, num_node)) # 对In的邻接矩阵进行归一化
    Out = normalize_digraph(edge2mat(outward, num_node)) # 对Out的邻接矩阵进行归一化
    # A = np.stack((I, In, Out))
    return I,In,Out

# 归一化邻接矩阵A
def normalize_adjacency_matrix(A):
    # 计算每一行的元素之和，存储在node——degrees中
    node_degrees = A.sum(-1)
    # 计算每个节点度数的倒数的平方根
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    # 接下来创建一个对角矩阵，对角线上元素是  degs_inv_sqrt
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    # 代码通过矩阵乘法操作 (norm_degs_matrix @ A @ norm_degs_matrix) 对邻接矩阵 A 进行归一化。结果被转换为 np.float32 类型，并返回。
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


# 计算邻接矩阵A的k阶邻接矩阵，用于更广泛的特征聚合
def k_adjacency(A, k, with_self=False, self_factor=1):
    # 确保A是一个Numpy数组
    assert isinstance(A, np.ndarray)
    # 创建一个大小与A相同的单位矩阵I
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    # 若k>0 则使用 np.linalg.matrix_power 函数计算 (A + I)^k 和 (A + I)^(k-1) 的幂，
    # 然后，代码使用 np.minimum 函数将 (A + I)^k 中的每个元素限制在 0 和 1 之间，再从中减去 (A + I)^(k-1) 中的每个元素限制在 0 和 1 之间，得到 k 阶邻接矩阵 Ak。
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)


    # 如果 with_self 参数为 True，则代码将自连接的权重加到 Ak 中，加权因子为 self_factor。
    # 最后，函数返回计算得到的 k 阶邻接矩阵 Ak。
    if with_self:
        Ak += (self_factor * I)
    return Ak


# 这段代码是用于生成多尺度空间图的函数。函数接受节点数量 num_node，自连接关系 self_link，内向边关系 inward，外向边关系 outward 作为输入，并返回生成的多尺度空间图。
def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    # 创建自连接的邻接矩阵
    I = edge2mat(self_link, num_node)
    # 创建内向边关系的邻接矩阵
    A1 = edge2mat(inward, num_node)
    # 创建外向边的邻接矩阵
    A2 = edge2mat(outward, num_node)
    # 创建内向边的二阶邻接矩阵
    A3 = k_adjacency(A1, 2)
    # 创建外向边的二阶邻接矩阵
    A4 = k_adjacency(A2, 2)
    # 归一化内向边关系的邻接矩阵
    A1 = normalize_digraph(A1)

    A2 = normalize_digraph(A2)  # 归一化外向边关系的邻接矩阵

    A3 = normalize_digraph(A3)  # 归一化内向边的二阶邻接矩阵

    A4 = normalize_digraph(A4)  # 归一化外向边的二阶邻接矩阵

    A = np.stack((I, A1, A2, A3, A4)) # 将邻接矩阵推叠成多尺度空间图
    return A


# 生成统一图谱
def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))  # 首先将neighbor和self link列表合并
    # 并调用edge2mat函数转化为邻接矩阵，并归一化后返回
    return A

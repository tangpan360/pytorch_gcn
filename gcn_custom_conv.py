import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNConvCustom(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConvCustom, self).__init__()
        # 初始化可学习的权重矩阵和偏置项
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x, edge_index):
        # 从边索引中提取行和列信息
        row, col = edge_index
        # 计算邻接矩阵A
        adj = torch.zeros((x.size(0), x.size(0)), device=x.device)
        adj[row, col] = 1  # 根据边索引设置邻接矩阵中的连接关系
        adj = torch.maximum(adj, adj.t())  # 确保邻接矩阵对称，适用于无向图
        
        # 添加自环连接（每个节点连接到自身）
        adj = adj + torch.eye(adj.size(0), device=x.device)

        # 对邻接矩阵进行对称归一化处理
        row_sum = adj.sum(1)  # 计算每个节点的度
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(row_sum + 1e-5))  # 计算度矩阵的逆平方根，添加小常数防止数值不稳定
        norm_adj = torch.matmul(D_inv_sqrt, torch.matmul(adj, D_inv_sqrt))  # 对称归一化公式：D^(-1/2) * A * D^(-1/2)

        # 执行图卷积操作
        out = torch.matmul(norm_adj, x)  # 聚合邻居信息：A * X
        out = torch.matmul(out, self.weight)  # 线性变换：A * X * W
        out = out + self.bias  # 添加偏置项

        return out

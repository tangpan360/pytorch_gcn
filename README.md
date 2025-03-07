## 引言
图卷积网络(Graph Convolutional Network)是处理图结构数据的核心算法。本文通过在Cora数据集上对比**原生实现**与**PyTorch Geometric框架实现**两种方案，解析GCN的关键技术细节。

## 项目结构
```
.
├── README.md
├── gcn_custom_conv.py
├── gcn_model_custom.py
├── gcn_model_torch_geometric.py
└── gcn_train_test.py
```

项目地址：[https://github.com/tangpan360/pytorch_gcn.git](https://github.com/tangpan360/pytorch_gcn.git)

## 一、代码结构解析
### 1.1 训练流程控制 (gcn_train_test.py)
```python
# gcn_train_test.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from gcn_model_custom import GCNCustom  # 自己实现的模型
from gcn_model_torch_geometric import GCN  # PyTorch Geometric提供的模型

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    # 前向传播
    out = model(data.x, data.edge_index)
    # 计算训练集上的损失
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # 反向传播
    loss.backward()
    optimizer.step()
    return loss

def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    # 使用训练掩码计算训练集准确率
    pred = out.argmax(dim=1)
    train_correct = pred[data.train_mask] == data.y[data.train_mask]
    train_acc = int(train_correct.sum()) / int(data.train_mask.sum())
    
    # 使用测试掩码计算测试集准确率
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    
    return train_acc, test_acc

def main():
    # 加载Cora数据集
    dataset = Planetoid(root='data/', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    # 选择模型：自己实现的GCN 或 PyTorch Geometric的GCN
    # model = GCNCustom(num_features=dataset.num_features, num_classes=dataset.num_classes)  # 自己实现的
    model = GCN(num_features=dataset.num_features, num_classes=dataset.num_classes)  # 使用PyTorch Geometric的

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 训练模型
    for epoch in range(200):
        loss = train(model, data, optimizer)
        train_acc, test_acc = test(model, data)
        if epoch % 1 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


if __name__ == '__main__':
    main()

```

### 1.2 框架实现版本 (gcn_model_torch_geometric.py)
```python
# gcn_model_torch_geometric.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # 引入PyTorch Geometric提供的GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16):
        super(GCN, self).__init__()
        # 使用PyTorch Geometric的图卷积层
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # 第一层：图卷积 + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 第二层：图卷积
        x = self.conv2(x, edge_index)
        
        # 输出层的softmax
        out = F.log_softmax(x, dim=1) 
        return out

```

PyG框架优势：
• 内置高效的稀疏矩阵运算
• 自动处理邻接矩阵归一化

### 1.3 原生实现版本 (gcn_model_custom.py + gcn_custom_conv.py)
```python
# gcn_custom_conv.py
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

```
```python
# gcn_model_custom.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # 引入PyTorch Geometric提供的GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16):
        super(GCN, self).__init__()
        # 使用PyTorch Geometric的图卷积层
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # 第一层：图卷积 + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 第二层：图卷积
        x = self.conv2(x, edge_index)
        
        # 输出层的softmax
        out = F.log_softmax(x, dim=1) 
        return out

```

核心技术点：
1. 邻接矩阵构建：从edge_index构建邻接矩阵
2. 自环添加：`adj = adj + torch.eye(...)`
3. 对称归一化：$D^{(-1/2)} AD^{(-1/2)}$
4. 特征传播：norm_adj @ x @ weight + bias

## 二、关键算法细节剖析
### 2.1 图卷积公式
GCN的核心公式：
$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$$

我们的实现对应：
```python
norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt  # 归一化
out = norm_adj @ x @ self.weight          # 特征变换
```

### 2.2 自环
```python
adj = adj + torch.eye(...)  # 添加自连接
```
• 保留节点自身特征

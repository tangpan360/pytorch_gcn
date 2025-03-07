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

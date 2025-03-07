import torch
import torch.nn.functional as F
from gcn_custom_conv import GCNConvCustom  # 引入我们自己实现的图卷积层

class GCNCustom(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16):
        super(GCNCustom, self).__init__()
        # 使用自己实现的图卷积层
        self.conv1 = GCNConvCustom(num_features, hidden_channels)
        self.conv2 = GCNConvCustom(hidden_channels, num_classes)

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

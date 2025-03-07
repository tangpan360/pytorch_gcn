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

import torch
data = torch.load('data/homogeneous_graph.pt')

# 计算各集合节点数
train_nodes = sum(data.train_mask).item()
val_nodes = sum(data.val_mask).item()
test_nodes = sum(data.test_mask).item()

# 计算正负例
def count_pos_neg(y, mask):
    pos = sum(y[mask]).item()
    return pos, sum(mask).item() - pos

train_pos, train_neg = count_pos_neg(data.y, data.train_mask)
val_pos, val_neg = count_pos_neg(data.y, data.val_mask)
test_pos, test_neg = count_pos_neg(data.y, data.test_mask)

print(f"训练集: {train_nodes}节点 (正例:{train_pos}, 负例:{train_neg})")
print(f"验证集: {val_nodes}节点 (正例:{val_pos}, 负例:{val_neg})")
print(f"测试集: {test_nodes}节点 (正例:{test_pos}, 负例:{test_neg})")
print(f"特征维度: {data.x.shape[1]}, 总边数: {data.edge_index.shape[1]}")
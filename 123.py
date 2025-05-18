import pandas as pd
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, degree
from torch_sparse import transpose
from torch_geometric.nn import global_mean_pool
from rdkit import Chem
from torch_geometric.nn import PNAConv, global_mean_pool
from rdkit.Chem import rdmolfiles, rdmolops
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, roc_curve
import deepchem as dc
from torch_geometric.loader import DataLoader
from torch_geometric.nn import InstanceNorm
from utils import evaluate
import copy
import sklearn.metrics as m
from torch_geometric.data import Batch
import random

from model.gsat import GSAT, ExtractorMLP
from model.pna import PNA

def eval_classification(labels, logits):
    auc = roc_auc_score(y_true=labels, y_score=logits)
    p, r, t = precision_recall_curve(y_true=labels, probas_pred=logits)
    aupr = m.auc(r, p)
    fpr, tpr, threshold = roc_curve(labels, logits)
    # 利用Youden's index计算阈值
    spc = 1 - fpr
    j_scores = tpr - fpr
    best_youden, youden_thresh, youden_sen, youden_spc = sorted(zip(j_scores, threshold, tpr, spc))[-1]
    predicted_label = copy.deepcopy(logits)
    youden_thresh = round(youden_thresh, 3)
    print(youden_thresh)

    predicted_label = [1 if i >= youden_thresh else 0 for i in predicted_label]
    p_1 = evaluate.precision(y_true=labels, y_pred=predicted_label)
    r_1 = evaluate.recall(y_true=labels, y_pred=predicted_label)
    acc = accuracy_score(y_true=labels, y_pred=predicted_label)
    f1 = f1_score(y_true=labels, y_pred=predicted_label)
    return p_1, r_1, acc, auc, aupr, f1


# === 药物分子图构建 ===
def smiles_to_graph(smiles):
    """ 使用DeepChem的ConvMolFeaturizer将SMILES转换为分子图表示 """
    featurizer = dc.feat.ConvMolFeaturizer()
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError(f"无法解析SMILES: {smiles}")

    mol_f = featurizer.featurize(mol)[0]  # 获取特征化后的分子
    
    # 获取原子特征
    x = torch.tensor(mol_f.get_atom_features(), dtype=torch.float)
    
    # 获取分子图的邻接列表
    edge_index_list = mol_f.get_adjacency_list()
    row, col = [], []
    for src, neighbors in enumerate(edge_index_list):
        for dst in neighbors:
            row.append(src)
            col.append(dst)
    
    edge_index = torch.tensor([row, col], dtype=torch.long)
    # edge_attr = torch.ones(edge_index.size(1), 1)  # 添加虚拟边特征
    
    return Data(x=x, edge_index=edge_index)

def get_cell_line_subgraph(cell_line_id, cpi, ppi, max_neighbors=50):
    related_proteins = set(cpi[cpi['cell'] == cell_line_id]['protein'].tolist())
    sub_ppi = ppi[(ppi['protein_a'].isin(related_proteins)) & (ppi['protein_b'].isin(related_proteins))]

    protein_to_idx = {protein: idx for idx, protein in enumerate(related_proteins)}
    
    # 创建原始边索引
    edge_list = [[protein_to_idx[p1], protein_to_idx[p2]] for p1, p2 in zip(sub_ppi['protein_a'], sub_ppi['protein_b'])]

    # 限制每个节点最多 20 个邻居
    neighbors = {}  # 记录每个节点的邻接点
    for src, dst in edge_list:
        if src not in neighbors:
            neighbors[src] = []
        if dst not in neighbors:
            neighbors[dst] = []
        neighbors[src].append(dst)
        neighbors[dst].append(src)

    # 进行邻居数目限制
    filtered_edges = []
    for node, adj in neighbors.items():
        if len(adj) > max_neighbors:
            sampled_adj = random.sample(adj, max_neighbors)  # 随机选择 20 个邻居
        else:
            sampled_adj = adj
        filtered_edges.extend([[node, neighbor] for neighbor in sampled_adj])

    # 转换成 PyTorch Tensor
    edge_index = torch.tensor(filtered_edges, dtype=torch.long).t()

    num_nodes = len(related_proteins)
    x = torch.randn(num_nodes, 64, dtype=torch.float)  # 生成随机特征

    return Data(x=x, edge_index=edge_index)
   

# === 预测器 ===
class DrugSynergyPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DrugSynergyPredictor, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim*3, 64), nn.BatchNorm1d(64),
                                    nn.ReLU(), nn.Dropout(0.2))
        # self.layer2 = nn.Sequential(nn.Linear(64, 32), 
        #                             nn.BatchNorm1d(32),
        #                             nn.ReLU(), nn.Dropout(0.2))
        # self.layer3 = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        
        self.layer2 = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        
    def forward(self, drug1_feat, drug2_feat, cell_feat):
        # TODO: 可以对细胞系采用注意力机制降维
        combined = torch.cat((drug1_feat, drug2_feat, cell_feat), dim=1)
        res1 = self.layer1(combined)
        # res2 = self.layer2(res1)
        # return self.layer3(res2)
        return self.layer2(res1)

class ExpSynergy(nn.Module):
    def __init__(self, drug1_gsat, drug2_gsat, cell_gsat, predictor):
        super(ExpSynergy, self).__init__()
        self.drug1_gsat = drug1_gsat
        self.drug2_gsat = drug2_gsat
        self.cell_gsat = cell_gsat
        self.predictor = predictor

    def forward(self, drug1, drug2, cell, epoch, training=True):
        _, drug1_feat = self.drug1_gsat.forward_pass(drug1, epoch, training=training)
        _, drug2_feat = self.drug2_gsat.forward_pass(drug2, epoch, training=training)
        _, cell_feat = self.cell_gsat.forward_pass(cell, epoch, training=training)
        return self.predictor(drug1_feat, drug2_feat, cell_feat)

    
def compute_degree(data_list, num_classes=10):
    """计算数据列表中的度分布"""
    batched_data = Batch.from_data_list(data_list)
    d = degree(batched_data.edge_index[1], num_nodes=batched_data.num_nodes, dtype=torch.long)
    return torch.bincount(d, minlength=num_classes)

def load_data(dataset_name):
    data_dir = f"./data/{dataset_name}"
    
    # 读取数据
    synergy = pd.read_csv(f"{data_dir}/drug_combinations.csv")
    synergy = synergy[['drug1_db', 'drug2_db', 'cell', 'synergy']]
    synergy = synergy.rename(columns={'drug1_db': 'drug1', 'drug2_db': 'drug2'})
    synergy['synergy'] = synergy['synergy'].apply(lambda x: 1 if x > 0 else 0)

    cpi = pd.read_csv(f"{data_dir}/cell_protein.csv")
    ppi = pd.read_excel(f"{data_dir}/protein-protein_network.xlsx", engine="openpyxl")
    drug2smiles = pd.read_csv(f"{data_dir}/drugsmiles.csv")

    return synergy, cpi, ppi, drug2smiles

# 在 main 函数中添加早停机制
def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 读取数据
    dataset_name = "OncologyScreen"  # 或者 "DrugCombDB"
    synergy, cpi, ppi, drug2smiles = load_data(dataset_name)
    drug2smilesdict = drug2smiles.set_index(drug2smiles.columns[0])[drug2smiles.columns[1]].to_dict()

    # 划分训练集和测试集
    train_df, test_df = train_test_split(synergy, test_size=0.2, random_state=42)

    # 构建训练数据
    start_time = time.time()
    drug1_list, drug2_list, cell_list, train_set = [], [], [], []
    for _, row in train_df.iterrows():
        drug_data1 = smiles_to_graph(drug2smilesdict[row['drug1']])
        drug_data2 = smiles_to_graph(drug2smilesdict[row['drug2']])
        cell_data = get_cell_line_subgraph(row['cell'], cpi, ppi)
        synergy_label = torch.tensor([row['synergy']], dtype=torch.float)
        drug1_list.append(drug_data1)
        drug2_list.append(drug_data2)
        cell_list.append(cell_data)
        train_set.append((drug_data1, drug_data2, cell_data, synergy_label))
    end_time = time.time()
    print(f"[INFO] 数据预处理耗时: {end_time - start_time:.4f} 秒")

    # 计算度分布
    print('[INFO] Calculating degree...')
    drug_deg = compute_degree(drug1_list + drug2_list)
    cell_deg = compute_degree(cell_list)
    print(f"[INFO] Drug Degree Distribution: {drug_deg}")
    print(f"[INFO] Cell Degree Distribution: {cell_deg}")

    # 配置 PNA
    hidden_size = 32
    drug_pna_config = {
        'hidden_size': hidden_size,
        'n_layers': 2,
        'dropout_p': 0.2,
        'aggregators': ['mean', 'max', 'min', 'std'],
        'scalers': True,
        'deg': drug_deg
    }
    cell_pna_config = {
        'hidden_size': hidden_size,
        'n_layers': 2,
        'dropout_p': 0.2,
        'aggregators': ['mean', 'max', 'min', 'std'],
        'scalers': True,
        'deg': cell_deg
    }

    # 初始化 GSAT
    drug_gsat1 = GSAT(
        clf=PNA(x_dim=75, edge_attr_dim=0, num_class=1, multi_label=False, model_config=drug_pna_config),
        extractor=ExtractorMLP(hidden_size=hidden_size, learn_edge_att=True),
        criterion=nn.BCELoss(),
        optimizer=optim.Adam,
        learn_edge_att=True
    )
    drug_gsat2 = GSAT(
        clf=PNA(x_dim=75, edge_attr_dim=0, num_class=1, multi_label=False, model_config=drug_pna_config),
        extractor=ExtractorMLP(hidden_size=hidden_size, learn_edge_att=True),
        criterion=nn.BCELoss(),
        optimizer=optim.Adam,
        learn_edge_att=True
    )
    cell_gsat = GSAT(
        clf=PNA(x_dim=64, edge_attr_dim=0, num_class=1, multi_label=False, model_config=cell_pna_config),
        extractor=ExtractorMLP(hidden_size=hidden_size, learn_edge_att=True),
        criterion=nn.BCELoss(),
        optimizer=optim.Adam,
        learn_edge_att=True
    )

    # 初始化模型
    model = ExpSynergy(drug_gsat1, drug_gsat2, cell_gsat, DrugSynergyPredictor(hidden_size)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # 调整学习率和权重衰减
    criterion = nn.BCELoss()
    # 加载训练数据
    batch_size = 256
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    print("Start Training...")

    # 构建测试数据
    test_data = []
    for _, row in test_df.iterrows():
        drug_data1 = smiles_to_graph(drug2smilesdict[row['drug1']])
        drug_data2 = smiles_to_graph(drug2smilesdict[row['drug2']])
        cell_data = get_cell_line_subgraph(row['cell'], cpi, ppi)
        synergy_label = torch.tensor([row['synergy']], dtype=torch.float)
        test_data.append((drug_data1, drug_data2, cell_data, synergy_label))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 早停机制参数
    patience = 10  # 容忍的验证集性能不提升的 epoch 数
    best_auc = 0.0  # 记录最佳 AUC
    best_epoch = 0  # 记录最佳 epoch
    best_model_state = None  # 保存最佳模型状态
    no_improve_count = 0  # 记录验证集性能未提升的次数

    # 训练循环
    for epoch in range(200):
        t = time.time()
        epoch_loss = 0
        model.train()

        # 训练阶段
        for batch in train_loader:
            drug_data1_batch, drug_data2_batch, cell_data_batch, labels_batch = batch
            drug_data1_batch = drug_data1_batch.to(device)
            drug_data2_batch = drug_data2_batch.to(device)
            cell_data_batch = cell_data_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            pred = model(drug_data1_batch, drug_data2_batch, cell_data_batch, epoch)
            loss = criterion(pred, labels_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 测试阶段
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                drug_data1_batch, drug_data2_batch, cell_data_batch, labels_batch = batch
                drug_data1_batch = drug_data1_batch.to(device)
                drug_data2_batch = drug_data2_batch.to(device)
                cell_data_batch = cell_data_batch.to(device)
                labels_batch = labels_batch.to(device)
                pred = model(drug_data1_batch, drug_data2_batch, cell_data_batch, epoch, training=False)
                y_true.extend(labels_batch.cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())

        y_true = [item[0] for item in y_true]
        y_pred = [item[0] for item in y_pred]
        p, r, acc, auc, aupr, f1 = eval_classification(y_true, y_pred)

        # 打印当前 epoch 的性能
        print(f'Epoch: {epoch}, Test: Precision {p:.4f} | Recall {r:.4f} | Accuracy {acc:.4f} | AUC {auc:.4f} | AUPR {aupr:.4f} | F1 {f1:.4f}')

        # 早停机制
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())  # 保存最佳模型状态
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}. Best AUC: {best_auc:.4f} at epoch {best_epoch}.")
            break

    # 加载最佳模型状态并输出最终性能
    model.load_state_dict(best_model_state)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            drug_data1_batch, drug_data2_batch, cell_data_batch, labels_batch = batch
            drug_data1_batch = drug_data1_batch.to(device)
            drug_data2_batch = drug_data2_batch.to(device)
            cell_data_batch = cell_data_batch.to(device)
            labels_batch = labels_batch.to(device)
            pred = model(drug_data1_batch, drug_data2_batch, cell_data_batch, epoch, training=False)
            y_true.extend(labels_batch.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    y_true = [item[0] for item in y_true]
    y_pred = [item[0] for item in y_pred]
    p, r, acc, auc, aupr, f1 = eval_classification(y_true, y_pred)
    print(f"Final Test Performance: Precision {p:.4f} | Recall {r:.4f} | Accuracy {acc:.4f} | AUC {auc:.4f} | AUPR {aupr:.4f} | F1 {f1:.4f}")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Dict, List, Optional

class FinancialGraphBuilder:
    def __init__(self, file_path: str):
        """
        初始化金融图构建器
        
        参数:
            file_path: 预处理数据的CSV文件路径
        """
        self.df = pd.read_csv(file_path)
        self.cat_cols = ['COUNTY_PROV', 'COUNTY_CITY', 'COUNTY_DIST', 'NET_CODE']
        self.encoder = None
        # 为三种不同类型的客户创建不同的映射字典
        self.person_to_idx = {}
        self.institution_to_idx = {}
        self.product_to_idx = {}
        self.fund_to_idx = {}
        
    def _preprocess_data(self):
        """数据预处理"""
        # 确保所有类别特征都是字符串类型
        self.df[self.cat_cols] = self.df[self.cat_cols].astype(str)
        
    def _prepare_onehot_encoder(self):
        """准备one-hot编码器"""
        categories = {
            col: self.df[col].dropna().astype(str).unique().tolist()
            for col in self.cat_cols
        }
        
        self.encoder = OneHotEncoder(
            categories=[categories[col] for col in self.cat_cols],
            sparse_output=False,
            handle_unknown='ignore'
        )
        self.encoder.fit(self.df[self.cat_cols])
        
    def _create_node_mappings(self):
        """创建节点ID到索引的映射"""
        # 分别处理三种不同类型的客户
        person_df = self.df[self.df['CUST_TYPE'] == 0]
        institution_df = self.df[self.df['CUST_TYPE'] == 1]
        product_df = self.df[self.df['CUST_TYPE'] == 2]
        
        self.person_to_idx = {cust_id: idx for idx, cust_id in enumerate(person_df['CUST_ID'].unique())}
        self.institution_to_idx = {cust_id: idx for idx, cust_id in enumerate(institution_df['CUST_ID'].unique())}
        self.product_to_idx = {cust_id: idx for idx, cust_id in enumerate(product_df['CUST_ID'].unique())}
        self.fund_to_idx = {fund: idx for idx, fund in enumerate(self.df['FUND_CODE'].unique())}
    
    def _get_person_features(self, cust_data: pd.Series) -> List[float]:
        """获取个人客户特征"""
        features = [
            float(cust_data['RISK_LEV']),
            float(cust_data['TELL_PREFIX'])
        ]
        
        # one-hot编码特征
        onehot = self.encoder.transform(
            pd.DataFrame([cust_data[self.cat_cols]], columns=self.cat_cols)
        )[0]
        features.extend(onehot)
        
        # 个人特有特征
        features.extend([
            float(cust_data['AGE']),
            float(cust_data.get('GENDER', 0))  # 默认为0如果GENDER缺失
        ])
            
        return features
    
    def _get_institution_features(self, cust_data: pd.Series) -> List[float]:
        """获取机构客户特征"""
        features = [
            float(cust_data['RISK_LEV']),
            float(cust_data['TELL_PREFIX'])
        ]
        
        # one-hot编码特征
        onehot = self.encoder.transform(
            pd.DataFrame([cust_data[self.cat_cols]], columns=self.cat_cols)
        )[0]
        features.extend(onehot)
        
        # 机构特有特征（如果有）
        # 这里可以添加机构特有的特征
        features.extend([0.0, 0.0])  # 填充保持特征维度一致
            
        return features
    
    def _get_product_features(self, cust_data: pd.Series) -> List[float]:
        """获取产品客户特征"""
        features = [
            float(cust_data['RISK_LEV']),
            float(cust_data['TELL_PREFIX'])
        ]
        
        # one-hot编码特征
        onehot = self.encoder.transform(
            pd.DataFrame([cust_data[self.cat_cols]], columns=self.cat_cols)
        )[0]
        features.extend(onehot)
        
        # 产品特有特征（如果有）
        features.extend([0.0, 0.0])  # 填充保持特征维度一致
            
        return features
    
    def _add_customer_nodes(self, data: HeteroData):
        """添加三种类型的客户节点到图中"""
        # 个人客户
        person_features = []
        for cust_id in self.person_to_idx:
            cust_data = self.df[self.df['CUST_ID'] == cust_id].iloc[0]
            features = self._get_person_features(cust_data)
            person_features.append(features)
            
        if person_features:
            person_features = np.array(person_features, dtype=np.float32)
            data['person'].x = torch.from_numpy(person_features)
        
        # 机构客户
        institution_features = []
        for cust_id in self.institution_to_idx:
            cust_data = self.df[self.df['CUST_ID'] == cust_id].iloc[0]
            features = self._get_institution_features(cust_data)
            institution_features.append(features)
            
        if institution_features:
            institution_features = np.array(institution_features, dtype=np.float32)
            data['institution'].x = torch.from_numpy(institution_features)
        
        # 产品客户
        product_features = []
        for cust_id in self.product_to_idx:
            cust_data = self.df[self.df['CUST_ID'] == cust_id].iloc[0]
            features = self._get_product_features(cust_data)
            product_features.append(features)
            
        if product_features:
            product_features = np.array(product_features, dtype=np.float32)
            data['product'].x = torch.from_numpy(product_features)
    
    def _add_fund_nodes(self, data: HeteroData):
        """添加基金节点到图中"""
        data['fund'].x = torch.arange(len(self.fund_to_idx)).view(-1, 1).float()
    
    def _add_edges(self, data: HeteroData):
        """添加边到图中"""
        # 需要为三种不同类型的客户分别创建边
        edge_indices = {'person': [], 'institution': [], 'product': []}
        edge_attributes = {'person': [], 'institution': [], 'product': []}  # 改名为edge_attributes避免冲突
        
        for _, row in self.df.iterrows():
            cust_type = row['CUST_TYPE']
            dst = self.fund_to_idx[row['FUND_CODE']]
            
            edge_attr = [
                row['BUSI_CODE'],
                row['CONF_AMTS'],
                row['CONF_YEAR'],
                row['CONF_MONTH'],
                row['CONF_DAY']
            ]
            
            if cust_type == 0:  # 个人
                src = self.person_to_idx[row['CUST_ID']]
                edge_indices['person'].append([src, dst])
                edge_attributes['person'].append(edge_attr)
            elif cust_type == 1:  # 机构
                src = self.institution_to_idx[row['CUST_ID']]
                edge_indices['institution'].append([src, dst])
                edge_attributes['institution'].append(edge_attr)
            elif cust_type == 2:  # 产品
                src = self.product_to_idx[row['CUST_ID']]
                edge_indices['product'].append([src, dst])
                edge_attributes['product'].append(edge_attr)
        
        # 添加三种不同类型的边
        if edge_indices['person']:
            edge_index = torch.tensor(edge_indices['person'], dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attributes['person'], dtype=torch.float)
            data['person', 'transaction', 'fund'].edge_index = edge_index
            data['person', 'transaction', 'fund'].edge_attr = edge_attr
            
        if edge_indices['institution']:
            edge_index = torch.tensor(edge_indices['institution'], dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attributes['institution'], dtype=torch.float)
            data['institution', 'transaction', 'fund'].edge_index = edge_index
            data['institution', 'transaction', 'fund'].edge_attr = edge_attr
            
        if edge_indices['product']:
            edge_index = torch.tensor(edge_indices['product'], dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attributes['product'], dtype=torch.float)
            data['product', 'transaction', 'fund'].edge_index = edge_index
            data['product', 'transaction', 'fund'].edge_attr = edge_attr
    
    def _add_targets(self, data: HeteroData):
        """添加目标标签(如果有)"""
        if 'target' in self.df.columns:
            # 为三种不同类型的客户分别添加目标
            for cust_type, node_type in [(0, 'person'), (1, 'institution'), (2, 'product')]:
                cust_ids = self.df[self.df['CUST_TYPE'] == cust_type]['CUST_ID'].unique()
                if len(cust_ids) > 0:
                    targets = self.df[self.df['CUST_ID'].isin(cust_ids)].groupby('CUST_ID')['target'].first().values
                    data[node_type].y = torch.from_numpy(targets).long()
    
    def build_graph(self) -> HeteroData:
        """
        构建异构图数据
        
        返回:
            构建好的HeteroData图对象
        """
        # 1. 数据预处理
        self._preprocess_data()
        
        # 2. 准备编码器
        self._prepare_onehot_encoder()
        
        # 3. 创建图数据对象
        data = HeteroData()
        
        # 4. 创建节点映射
        self._create_node_mappings()
        
        # 5. 添加节点
        self._add_customer_nodes(data)
        self._add_fund_nodes(data)
        
        # 6. 添加边
        self._add_edges(data)
        
        # 7. 添加目标标签(如果有)
        self._add_targets(data)
        
        return data
    
    def validate_graph(self, data: HeteroData):
        """验证图数据结构并打印样本数据"""
        print("\n=== 图结构基本信息 ===")
        print(data)
        print("\n节点类型:", data.node_types)
        print("\n边类型:", data.edge_types)
        
        # 打印各类节点的特征维度
        print("\n=== 节点特征维度 ===")
        for node_type in ['person', 'institution', 'product', 'fund']:
            if node_type in data.node_types:
                print(f"{node_type}节点特征维度:", data[node_type].x.shape)
        
        # 打印各类边的特征维度
        print("\n=== 边特征维度 ===")
        for edge_type in data.edge_types:
            print(f"边{edge_type}特征维度:", data[edge_type].edge_attr.shape if hasattr(data[edge_type], 'edge_attr') else "无特征")
        
        # 打印样本节点数据
        print("\n=== 样本节点数据 ===")
        for node_type in ['person', 'institution', 'product', 'fund']:
            if node_type in data.node_types and hasattr(data[node_type], 'x'):
                print(f"\n{node_type}节点样本数据 (前5个):")
                print(data[node_type].x[:5])
                if hasattr(data[node_type], 'y'):
                    print(f"{node_type}节点标签样本 (前5个):")
                    print(data[node_type].y[:5])
        
        # 打印样本边数据
        print("\n=== 样本边数据 ===")
        for edge_type in data.edge_types:
            print(f"\n边类型 {edge_type} 样本数据:")
            print("边索引 (前5条):")
            print(data[edge_type].edge_index[:, :5])
            if hasattr(data[edge_type], 'edge_attr'):
                print("边特征 (前5条):")
                print(data[edge_type].edge_attr[:5])
        
        # 打印节点数量统计
        print("\n=== 节点数量统计 ===")
        for node_type in data.node_types:
            print(f"{node_type}节点数量:", data[node_type].x.shape[0])
        
        # 打印边数量统计
        print("\n=== 边数量统计 ===")
        for edge_type in data.edge_types:
            num_edges = data[edge_type].edge_index.shape[1]
            print(f"边类型 {edge_type} 数量:", num_edges)
# 使用示例
if __name__ == "__main__":
    # 创建图构建器
    builder = FinancialGraphBuilder('data/preprocessed_data_gnn.csv')
    
    # 构建图数据
    graph_data = builder.build_graph()
    
    # 验证图数据
    builder.validate_graph(graph_data)
    
    # 保存图数据 (可选)
    torch.save(graph_data, 'financial_graph_3.pt')
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict

def build_heterogeneous_graph(df):
    """构建异构图数据的完整流程
    
    Args:
        df (pd.DataFrame): 预处理后的原始数据
        
    Returns:
        HeteroData: 构建好的异构图数据
    """
    # ========== 初始化 ==========
    data = HeteroData()
    
    # 初始化编码器
    location_encoder = LabelEncoder()
    gender_encoder = LabelEncoder()
    net_encoder = LabelEncoder()
    risk_encoder = LabelEncoder()
    age_scaler = StandardScaler()

    # ========== Step 1: 构建节点 ==========
    node_maps = {
        "person": defaultdict(int),
        "institution": defaultdict(int),
        "product": defaultdict(int),
        "fund": defaultdict(int)
    }

    node_features = {
        "person": {"gender": [], "age": [], "location": [], "tell": [], "net": [], "risk": []},
        "institution": {"location": [], "tell": [], "net": [], "risk": []},
        "product": {"location": [], "tell": [], "net": [], "risk": []},
        "fund": {}
    }

    # 处理地理信息
    df["location"] = (
        df["COUNTY_PROV"].fillna(-1).astype(str) + "_" + 
        df["COUNTY_CITY"].fillna(-1).astype(str) + "_" + 
        df["COUNTY_DIST"].fillna(-1).astype(str)
    )

    # 构建节点映射和特征
    for _, row in df.iterrows():
        # 处理账户节点
        cust_type = ["person", "institution", "product"][row["CUST_TYPE"]]
        cust_id = row["CUST_ID"]
        
        if cust_id not in node_maps[cust_type]:
            node_maps[cust_type][cust_id] = len(node_maps[cust_type])
            
            # 收集特征
            if cust_type == "person":
                node_features["person"]["gender"].append(row["GENDER"])
                node_features["person"]["age"].append(row["AGE"])
                node_features["person"]["location"].append(row["location"])  # 使用小写
                node_features["person"]["tell"].append(row["TELL_PREFIX"])
                node_features["person"]["net"].append(row["NET_CODE"])
                node_features["person"]["risk"].append(row["RISK_LEV"])
            else:
                node_features[cust_type]["location"].append(row["location"])  # 使用小写
                node_features[cust_type]["tell"].append(row["TELL_PREFIX"])
                node_features[cust_type]["net"].append(row["NET_CODE"])
                node_features[cust_type]["risk"].append(row["RISK_LEV"])

        # 处理基金节点
        fund_code = row["FUND_CODE_FREQ"]
        if fund_code not in node_maps["fund"]:
            node_maps["fund"][fund_code] = len(node_maps["fund"])

    # ========== Step 2: 构建边 ==========
    edge_data = defaultdict(lambda: {"src": [], "dst": [], "amount": [], "time": []})

    for _, row in df.iterrows():
        cust_type = ["person", "institution", "product"][row["CUST_TYPE"]]
        src = node_maps[cust_type][row["CUST_ID"]]
        dst = node_maps["fund"][row["FUND_CODE_FREQ"]]
        edge_type = (cust_type, f"busi_{row['BUSI_CODE']}", "fund")
        
        edge_data[edge_type]["src"].append(src)
        edge_data[edge_type]["dst"].append(dst)
        edge_data[edge_type]["amount"].append(row["CONF_AMTS"])
        edge_data[edge_type]["time"].append(f"{row['CONF_YEAR']}-{row['CONF_MONTH']}-{row['CONF_DAY']}")

    # ========== Step 3: 特征编码 ==========
    # 合并所有特征用于编码器训练
    all_locations = (
        node_features["person"]["location"] + 
        node_features["institution"]["location"] + 
        node_features["product"]["location"]
    )
    location_encoder.fit(all_locations)

    all_nets = (
        node_features["person"]["net"] + 
        node_features["institution"]["net"] + 
        node_features["product"]["net"]
    )
    net_encoder.fit(all_nets)

    all_risks = (
        node_features["person"]["risk"] + 
        node_features["institution"]["risk"] + 
        node_features["product"]["risk"]
    )
    risk_encoder.fit(all_risks)

    gender_encoder.fit(node_features["person"]["gender"])
    age_scaler.fit([[age] for age in node_features["person"]["age"]])

    # ========== Step 4: 构建特征矩阵 ==========
    def get_features(features, node_type):
        """为指定节点类型构建特征矩阵"""
        if node_type == "person":
            return torch.tensor([
                [
                    gender_encoder.transform([g])[0],
                    age_scaler.transform([[a]])[0][0],
                    location_encoder.transform([l])[0],
                    net_encoder.transform([n])[0],
                    risk_encoder.transform([r])[0]
                ]
                for g, a, l, n, r in zip(
                    features["gender"],
                    features["age"],
                    features["location"],
                    features["net"],
                    features["risk"]
                )
            ], dtype=torch.float)
        elif node_type in ["institution", "product"]:
            return torch.tensor([
                [
                    location_encoder.transform([l])[0],
                    net_encoder.transform([n])[0],
                    risk_encoder.transform([r])[0]
                ]
                for l, n, r in zip(
                    features["location"],
                    features["net"],
                    features["risk"]
                )
            ], dtype=torch.float)
        else:  # fund节点
            return torch.randn(len(node_maps["fund"]), 8)

    # 添加节点特征
    for node_type in ["person", "institution", "product", "fund"]:
        data[node_type].x = get_features(node_features.get(node_type, {}), node_type)

    # ========== Step 5: 添加边 ==========
    for edge_type, attributes in edge_data.items():
        if not attributes["src"]:
            continue
            
        data[edge_type].edge_index = torch.tensor(
            [attributes["src"], attributes["dst"]], dtype=torch.long
        )
        data[edge_type].edge_attr = torch.tensor(
            attributes["amount"], dtype=torch.float
        ).view(-1, 1)
        
        if "time" in attributes:
            data[edge_type].time = attributes["time"]

    return data


# ========== 使用示例 ==========
if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed_data_ml.csv")
    graph_data = build_heterogeneous_graph(df)
    print("构建的图数据结构:")
    print(graph_data)
    print("\n节点数量统计:", {k: v.x.size(0) for k, v in graph_data.node_items()})
    print(graph_data['person'].x[1])
    # print(graph_data['person','busi_1','fund'].edge_index[0])
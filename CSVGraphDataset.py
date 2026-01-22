import numpy as np
import pandas as pd
import os
import torch

from torch_geometric.data import InMemoryDataset, Data


class CSVGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def graph_file_names(self):
        return ['edges.csv', 'nodes.csv', 'properties.csv', 'tradition.csv']

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    def process(self):
        edges = pd.read_csv(os.path.join(self.raw_dir, 'edges.csv'))
        nodes = pd.read_csv(os.path.join(self.raw_dir, 'nodes.csv'))
        properties = pd.read_csv(os.path.join(self.raw_dir, 'properties.csv'))
        trad_features = pd.read_csv(os.path.join(self.raw_dir, 'tradition.csv' ))

        # delete label
        exclude_columns = ['File', 'HeuBug', 'HeuBugCount', 'RealBug', 'RealBugCount']

        trad_columns = [col for col in trad_features.columns if col not in exclude_columns]

        label_dict = {
            row["graph_id"]: 1 if row["label"] >= 1 else 0
            for _, row in properties.iterrows()
        }
        num_nodes_dict = {
            row["graph_id"]: row["num_nodes"]
            for _, row in properties.iterrows()
        }

        class_to_graph_id = {
            row["class_name"]: row["graph_id"]
            for _, row in properties.iterrows()
        }

        tradition_dict = {}

        for _, row in trad_features.iterrows():
            file_name = row["File"]
            file_name = file_name[:-5]
            tradition_dict[file_name] = {
                col: row[col] for col in trad_columns
            }

        graph_tradition_dict = {}
        for file_name, trad_metrics in tradition_dict.items():
            if file_name in class_to_graph_id:
                graph_id = class_to_graph_id[file_name]
                graph_tradition_dict[graph_id] = trad_metrics
            else:
                print(f"警告: 文件 {file_name} 在properties.csv中找不到对应的class_name")

        edges_group = edges.groupby("graph_id")
        nodes_group = nodes.groupby("graph_id")

        data_list = []

        for graph_id in edges_group.groups:
            if graph_id not in nodes_group.groups:
                continue

            edge_df = edges_group.get_group(graph_id)
            node_df = nodes_group.get_group(graph_id).sort_values(by="node_id")

            node_ids = sorted(node_df["node_id"].unique())
            node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_ids)}

            edge_df["src"] = edge_df["src"].map(node_id_map)
            edge_df["dst"] = edge_df["dst"].map(node_id_map)

            node_df["node_id"] = node_df["node_id"].map(node_id_map)

            num_nodes = num_nodes_dict[graph_id]

            label = label_dict[graph_id]

            src = edge_df["src"].to_numpy()
            dst = edge_df["dst"].to_numpy()
            assert np.all(src < num_nodes), "src索引超出范围"
            assert np.all(dst < num_nodes), "dst索引超出范围"
            edge_index = torch.tensor([src, dst], dtype=torch.long)

            deg_in = np.zeros(num_nodes, dtype=np.float32)
            deg_out = np.zeros(num_nodes, dtype=np.float32)
            for s in src:
                if s < num_nodes:
                    deg_out[s] += 1
            for d in dst:
                if d < num_nodes:
                    deg_in[d] += 1
            deg_in = torch.tensor(deg_in).view(-1, 1)
            deg_out = torch.tensor(deg_out).view(-1, 1)

            if graph_id in graph_tradition_dict:
                trad_metrics = graph_tradition_dict[graph_id]
                print(f"图 {graph_id} 成功匹配传统度量特征")
            else:
                print(f"[wraning]: 图 {graph_id} 未成功匹配传统度量特征")

            trad_features = []
            for col in trad_columns:
                trad_features.extend([trad_metrics[col]] * num_nodes)

            trad_tensor = torch.tensor(trad_features, dtype=torch.float32).view(num_nodes, len(trad_columns))

            x = torch.cat([deg_in, deg_out, trad_tensor], dim=1)

            y = torch.tensor([label],dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)

            data_list.append(data)

            print('graph_id : {} Created Successfully'.format(graph_id))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

from utilities.utils import MyUtils


class GCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_dim = args['in_dim']
        self.output_dim = args['output_dim']
        self.epochs = args['epochs']
        self.device = args['device']

        self.hidden_dim = args['hidden_dim']
        self.num_layers = args['n_layers']
        self.lr = args['lr']
        self.weight_decay = args['weight_decay']
        self.dropout = args['dropout']

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(self.in_dim, self.hidden_dim))
        for _ in range(1, self.num_layers):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

        # Global attention pooling
        self.att_pool = GlobalAttention(
            gate_nn=nn.Linear(self.hidden_dim, 1)
        )

        # Classification head
        self.linear = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.classify = nn.Linear(2 * self.hidden_dim, self.output_dim)

    def forward(self, x, edge_index, batch):
        hg = None
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Graph-level pooling
            pooled = self.att_pool(x, batch)

            if hg is None:
                hg = pooled
            else:
                # 累加并归一化，避免特征过大
                hg = hg + pooled

        # Classification head
        out = self.linear(hg)
        out = F.relu(F.dropout(out, p=self.dropout, training=self.training))
        out = self.classify(out)
        return out
    def train_GCN(self, model, dataloader, dataset):
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        weight = MyUtils.class_weight(dataset, self.device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        model.train()
        for epoch in range(self.epochs):
            for batch in dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()


    def predict_GCN(self, model, data):
        model.eval()
        preds = []
        trues = []
        for batch in data:
            batch = batch.to(self.device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(batch.y.cpu().numpy())
        return preds, trues

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GlobalAttention
from utilities.utils import MyUtils  # 可选：处理类别权重







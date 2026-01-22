import sys
import warnings
import numpy as np
import torch

from torch_geometric.loader import DataLoader
from CSVGraphDataset import CSVGraphDataset
from GNN.GCN import GCN
from GNN.GIN import GIN
from GNN.GAT import GAT
from GNN.GraphSAGE import GraphSAGE
from utilities.File import create_dir, save_results
from utilities.PerformanceMeasure import get_measure

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    if len(sys.argv) != 4:
        print("Usage: python GN_demo.py <trainProjectName> <trainProjectVersion> <testProjectName> <testProjectVersion> <modelType> <GraphType> <Reps>")
        print("Example: python GN_demo.py hbase 0.94.0 hbase 0.95.0 GAT CFG 30")

    trainProjectName = sys.argv[1]
    trainProjectVersion = sys.argv[2]
    testProjectName = sys.argv[3]
    testProjectVersion = sys.argv[4]
    modelType = sys.argv[5]
    GraphType = sys.argv[6]
    Reps = int(sys.argv[7])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = f'./out_CPDP/{GraphType}/{modelType}/'
    trainDataset = CSVGraphDataset(root=f'./data/graph/{trainProjectName}/{trainProjectVersion}/{GraphType}/').to(device)
    testDataset = CSVGraphDataset(root=f'./data/graph/{testProjectName}/{testProjectVersion}/{GraphType}/').to(device)

    in_dim = trainDataset.num_node_features

    args = {'lr': 1e-3, 'weight_decay': 1e-4, 'epochs': 500, 'batch_size': 64,
            'in_dim': in_dim, 'hidden_dim': 32, 'output_dim': 2,
            'n_layers': 4, 'dropout': 0.2, 'device': device, 'n_heads': 5}


    for Rep in range(Reps):
        print(f"------Rep {Rep+1}: {GraphType} {modelType} {trainProjectName}-{trainProjectVersion}to{testProjectName}-{testProjectVersion}  Start!!!------")
        train_loader = DataLoader(trainDataset, batch_size=args['batch_size'], shuffle=True)
        test_loader = DataLoader(testDataset, batch_size=args['batch_size'])
        if modelType == 'GCN':
            GNmodel = GCN(args).to(device)
            GNmodel.train_GCN(GNmodel, train_loader, trainDataset)
            predict_y, true_y = GNmodel.predict_GCN(GNmodel ,test_loader)
        elif modelType == 'GAT':
            GNmodel = GAT(args).to(device)
            GNmodel.train_GAT(GNmodel, train_loader, trainDataset)
            predict_y, true_y = GNmodel.predict_GAT(GNmodel, test_loader)
        elif modelType == 'GraphSAGE':
            GNmodel = GraphSAGE(args).to(device)
            GNmodel.train_SAGE(GNmodel, train_loader, trainDataset)
            predict_y, true_y = GNmodel.predict_SAGE(GNmodel ,test_loader)
        elif modelType == 'GIN':
            GNmodel = GIN(args).to(device)
            GNmodel.train_GIN(GNmodel, train_loader, trainDataset)
            predict_y, true_y = GNmodel.predict_GIN(GNmodel ,test_loader)

        predict_y = np.array(predict_y)

        precision, recall, pf, F1, AUC, g_measure, g_mean, bal, MCC = get_measure(true_y, predict_y)

        measure = [precision, recall, pf, F1, AUC, g_measure, g_mean, bal, MCC,]

        fres = create_dir(save_path)
        save_results(fres + f'results_add_{trainProjectName}_{trainProjectVersion}_to_{testProjectName}_{testProjectVersion}', measure)
        print("===========P=============\n{}\n".format(measure[0]))
        print("===========R=============\n{}\n".format(measure[1]))
        print("===========F1=============\n{}\n".format(measure[3]))
        print("===========AUC=============\n{}\n".format(measure[4]))
        print("===========MCC=============\n{}\n".format(measure[8]))


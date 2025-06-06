import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, EdgeConv, global_max_pool
from torch_geometric.data import Data, DataLoader

class DGCNN(torch.nn.Module):
    def __init__(self, k=20, emb_dims=1024, output_channels=40):
        super(DGCNN, self).__init__()
        self.k = k

        # EdgeConv layers
        self.conv1 = EdgeConv(self.mlp(6, 64), aggr='max')
        self.conv2 = EdgeConv(self.mlp(128, 64), aggr='max')
        self.conv3 = EdgeConv(self.mlp(128, 128), aggr='max')
        self.conv4 = EdgeConv(self.mlp(256, 256), aggr='max')

        self.lin1 = nn.Linear(512, emb_dims)
        self.bn1 = nn.BatchNorm1d(emb_dims)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(emb_dims, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, output_channels)

    def mlp(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, batch):
        pos = x[:, :3]

        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=False)
        x1 = self.conv1(x, edge_index)

        edge_index = knn_graph(x1, k=self.k, batch=batch, loop=False)
        x2 = self.conv2(x1, edge_index)

        edge_index = knn_graph(x2, k=self.k, batch=batch, loop=False)
        x3 = self.conv3(x2, edge_index)

        edge_index = knn_graph(x3, k=self.k, batch=batch, loop=False)
        x4 = self.conv4(x3, edge_index)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x = global_max_pool(x_cat, batch)

        x = F.relu(self.bn1(self.lin1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.fc1(x)))
        x = self.dp2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

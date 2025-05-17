from knn import knn_neighbors
from fps import fps
import fastmesh as fm
import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt

path_context = "/home/waleed/Documents/3DLearning/marginline/final/C010_47/47_margin_context.bmesh"
path_context2 = "/home/waleed/Documents/3DLearning/marginline/final/C005_36/36_margin_context.bmesh"
path_margin = "/home/waleed/Documents/3DLearning/marginline/final/C010_47/47_margin.pts"
path_margin2 = "/home/waleed/Documents/3DLearning/marginline/final/C005_36/36_margin.pts"

# read the margin points
with open(path_margin, "r") as f:
    lines = f.readlines()
    margin_points = np.array([list(map(float, line.split())) for line in lines[1:]])

vertices = fm.load(path_context)[0]
vertices2 = fm.load(path_context2)[0]
print("loaded", vertices.shape)
vertices = fps(vertices, 2048)[0]
vertices2 = fps(vertices2, 2048)[0]
print("fps", vertices.shape)
vertices = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0)
vertices2 = torch.tensor(vertices2, dtype=torch.float32).unsqueeze(0)
print("before ", vertices.shape)
edge_feats, neighbors = knn_neighbors(vertices, 32)
edge_feats2, neighbors2 = knn_neighbors(vertices2, 32)
print("edge_feats", edge_feats.shape)
print("neighbors", neighbors.shape)

ver = torch.stack([vertices, vertices2], dim=0).squeeze(1)
teeth = torch.tensor(np.array([10,12]), dtype=torch.long).view(-1)
print(ver.shape, "FFF")

device = 'cpu'
from dataloader import AtomicaMarginLine
from models.Pipeline import Model
from models.FoldingNet.FoldingNet import FoldingNet 
model = Model().to(device)
foldNET = FoldingNet(num_points=400)
model.eval()
print(model(vertices, teeth).shape)
foldNET(vertices)

# train_loader, test_loader = AtomicaMarginLine(1024, 400, 2, 8)
# 
# for vertices, marginline, teeth in test_loader:
    # vertices, marginline, teeth = vertices.to(device), marginline.to(device), teeth.to(device)
    # print(vertices.shape, teeth.shape)
    # out = model(vertices, teeth)
    # break
# print(out.shape)


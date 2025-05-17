from fps import fps
import fastmesh as fm
import numpy as np
import torch

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
vertices = fps(vertices, 2048)[0]
vertices2 = fps(vertices2, 2048)[0]
vertices = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0)
vertices2 = torch.tensor(vertices2, dtype=torch.float32).unsqueeze(0)

ver = torch.stack([vertices, vertices2], dim=0).squeeze(1)
teeth = torch.tensor(np.array([10,12]), dtype=torch.long).view(-1)
print(ver.shape, "FFF")
print(vertices.shape, "f")

from models.FoldingNet.FoldingNet import FoldingNet 
foldNET = FoldingNet(num_points=20)
out = foldNET(ver)
print(out.shape)
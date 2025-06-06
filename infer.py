from scipy.interpolate import interp1d
import os
import torch
import numpy as np
import fastmesh as fm
import trimesh.scene
# from models.FoldingNet.FoldingNet2 import FoldingNet
from models.Gem_torch.EdgeConv import DGCNN_Generator
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter_mean
from ChamferLoss import ChamferLoss

from fps import fps
import trimesh
 

pretrained = "/home/waleed/Documents/3DLearning/margin-line/checkpoints/my_model_best_10000.pth"
info = np.load("/home/waleed/Documents/3DLearning/margin-line/final/context_margin_colors_faces_classes/C018_11.npz")
truemarginline = info['margin'] 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vertices = torch.tensor(fps(info['context'], 2048)[0],  dtype=torch.float32)
mean = vertices.mean() 
vertices -= mean
truemarginline -= mean.numpy()

teeth = torch.tensor(np.maximum(0, np.array(11) - 10 - 2 * ((np.array(11) // 10) - 1)), dtype=torch.long)
# model = FoldingNet(num_points=20)
model = DGCNN_Generator(num_output_points=400).to(device)

# Load pretrained weights if provided
state_dict = torch.load(pretrained, map_location=device)
model.load_state_dict(state_dict)
print(teeth.shape)
model.eval()
# data = Data(pos=vertices, batch = 0, y=truemarginline, tooth_n=teeth.view(1)).to(device)

def _load_marginline(marginline):
    """Load margin line from .pts file."""

    N = marginline.shape[0]
    if N > 400:
        marginline = fps(marginline, 400, h=7)[0]

    elif N < 400:
        closed_margin = np.vstack([marginline, marginline[0]])

        diffs = np.diff(closed_margin, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        arc_lengths = np.concatenate([[0], np.cumsum(dists)])

        arc_lengths /= arc_lengths[-1]

        t_target = np.linspace(0, 1, 400, endpoint=False)

        interpolated = []
        for dim in range(3):
            interp_func = interp1d(arc_lengths, closed_margin[:, dim], kind='linear', assume_sorted=True)
            interpolated.append(interp_func(t_target))

        marginline = np.stack(interpolated, axis=1)
    return marginline

truemarginline = _load_marginline(truemarginline)
data = Data(
    pos=vertices,
    batch=torch.zeros(vertices.shape[0], dtype=torch.long),  # All points in batch 0
    y=torch.tensor(truemarginline, dtype=torch.float32),
    tooth_n=teeth.view(1)
).to(device)

def center_data(data, batch_size=1, num_points=400):
    # Compute per-sample mean using the batch vector
    mean = scatter_mean(data.pos, data.batch, dim=0)
    batch_y = torch.arange(batch_size).repeat_interleave(num_points).to(device)
    data.pos -= mean[data.batch]
    data.y -= mean[batch_y]
    return data
data = center_data(data)

marginline = model(data).squeeze(0)

cloud1 = trimesh.points.PointCloud(vertices.cpu().numpy(), colors=[255, 0, 0])
cloud2 = trimesh.points.PointCloud(truemarginline, colors=[0, 255, 0])
cloud3 = trimesh.points.PointCloud(marginline.detach().cpu().numpy(), colors=[0, 0, 255])

scene = trimesh.scene.Scene([cloud1, cloud3])
scene.show()

loss = ChamferLoss()(marginline.view(1, 400, 3), data.y.view(1, 400, 3))

print("Chamfer Loss:", loss.item())
import torch
import torch.nn as nn
from models.GraphCNN.DGCNN import EdgeConv

class DGCNN(nn.Module):
    def __init__(self):
        super(DGCNN, self).__init__()

        self.embedding = nn.Embedding(32, 64)
        self.conv_emb = nn.Conv1d(64, 64, 1)

        self.edgeconv1 = EdgeConv(3, [64, 64])
        self.edgeconv2 = EdgeConv(64, [64, 64])
        self.edgeconv3 = EdgeConv(64, [64])

        self.conv1 = nn.Conv1d(195, 256, 1) # 1024
        self.conv2 = nn.Conv1d(515, 128, 1) # 1283, 128 ## 1024

        self.relu = nn.LeakyReLU()

    def forward(self, x, jaw=0):  ### B, N, Cin -> # B, Nout, Cout
        emb = self.embedding(jaw).unsqueeze(2)
        
        emb = self.conv_emb(emb).permute(0, 2, 1)

        x1 = self.edgeconv1(x)
        x2 = self.edgeconv2(x1)
        x3 = self.edgeconv3(x2)

        xn = torch.cat([x, x1, x2, x3], dim=2).permute(0, 2, 1)
        xn = torch.max(xn, dim = 2, keepdim=True)[0]
        xn = self.relu(self.conv1(xn)).permute(0, 2, 1)
        xn = torch.cat([xn, emb], dim=2)
        x = torch.cat([xn.expand(x1.size(0), x1.size(1), -1), x1, x2, x3, x], dim=2).permute(0, 2, 1)

        x = self.relu(self.conv2(x)).transpose(1, 2)

        return x

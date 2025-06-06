import torch
from torch import nn
from torch_geometric.nn import DynamicEdgeConv, global_max_pool

class DGCNN_Generator(torch.nn.Module):
    def __init__(self, k=10, emb_dims=512, num_output_points=1024):
        super().__init__()
        self.k = k
        self.num_output_points = num_output_points

        self.embedding = nn.Embedding(33, 64)
        self.conv_emb = nn.Conv1d(64, 64, 1)

        # Encoder: DGCNN EdgeConv layers
        self.conv1 = DynamicEdgeConv(nn=torch.nn.Sequential(
            torch.nn.Linear(6, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64)), k=k)

        self.conv2 = DynamicEdgeConv(nn=torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64)), k=k)

        self.conv3 = DynamicEdgeConv(nn=torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64)), k=k)

        self.conv4 = DynamicEdgeConv(nn=torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)), k=k)

        # Latent representation per cloud
        self.encoder_fc = torch.nn.Sequential(
            torch.nn.Linear(384, emb_dims),
            torch.nn.ReLU()
        )

        # Decoder: map global feature to new point cloud
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(emb_dims, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, num_output_points * 3)
        )

    def forward(self, data):
        # print(torch.isnan(data.pos).any(), torch.isinf(data.pos).any())
        # print(torch.isnan(data.batch).any(), torch.isinf(data.batch).any())
        # print(data.batch.min(), data.batch.max())
        x, batch, tooth_n = data.pos, data.batch, data.tooth_n
        # batch_size, num_points, _ = x.shape
        # x = x.view(-1, 3)
        # batch = torch.arange(batch_size).repeat_interleave(num_points)
        emb = self.embedding(tooth_n)
        print(emb.shape)
        print(tooth_n.shape)
        emb = self.conv_emb(emb.unsqueeze(2))
        print(emb.shape)
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)
        print(x1.shape, x2.shape, x3.shape, x4.shape, emb.squeeze(2)[batch].shape, batch)
        x = torch.cat((x1, x2, x3, x4, emb.squeeze(2)[batch]), dim=1)
        x = global_max_pool(x, batch)  # [B, 364]

        x = self.encoder_fc(x)         # [B, emb_dims]        
        x = self.decoder(x)            # [B, num_output_points * 3]

        x = x.view(-1, self.num_output_points, 3)  # Reshape to [B, N_out, 3]
        return x

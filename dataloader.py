from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data

import os
import numpy as np
from fps import fps


class MarginLineDataset(Dataset):
    def __init__(self, split='train', transform=None, args=None):
        """
        Args:
            root_dir (string): Directory with all the parts (data_part_{1-6}).
            split (string): 'train' or 'test' to select the appropriate dataset.
            test_ids_file (string): Path to the txt file containing IDs for testing.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.split = split
        self.cases_path = f'{split}.txt'
        self.data_dir = args.path
        self.centroids = args.centroids
        self.marginNum = args.marginNum
        self.transform = transform
        self.cases = self._load_cases()
        self.data_list = self._prepare_data_list()

    def _load_cases(self):
        """Load IDs from private-testing-set.txt."""
        with open(self.cases_path, 'r') as f:
            cases = [line.strip() for line in f.readlines()]
        return cases

    def _prepare_data_list(self):
        """Prepare the list of data paths for training or testing."""
        data_list = []
        for case_id in self.cases:
            new_case_path = os.path.join(self.data_dir, case_id)
            # Load the .bmesh and .pts files
            tooth_id = case_id.split('.')[0][-2:]
            tooth_enc = np.maximum(0, int(tooth_id) - 10 - 2 * ((int(tooth_id) // 10) - 1))

            data_list.append((new_case_path, tooth_enc))
        return data_list

    def __len__(self):
        return len(self.data_list)
 
    def _load_marginline(self, marginline):
        """Load margin line from .pts file."""

        N = marginline.shape[0]
        if N > self.marginNum:
            marginline = fps(marginline, self.marginNum, h=7)[0]

        elif N < self.marginNum:
            closed_margin = np.vstack([marginline, marginline[0]])

            diffs = np.diff(closed_margin, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            arc_lengths = np.concatenate([[0], np.cumsum(dists)])

            arc_lengths /= arc_lengths[-1]

            t_target = np.linspace(0, 1, self.marginNum, endpoint=False)

            interpolated = []
            for dim in range(3):
                interp_func = interp1d(arc_lengths, closed_margin[:, dim], kind='linear', assume_sorted=True)
                interpolated.append(interp_func(t_target))

            marginline = np.stack(interpolated, axis=1)
        return marginline

    def __getitem__(self, idx):
        path, tooth_n = self.data_list[idx]
        info = np.load(path)
        vertices_np = info["context"]
        marginline = info["margin"]

        vertices = fps(vertices_np, self.centroids)[0]
        marginline = self._load_marginline(marginline)

        # Convert vertices to a PyTorch tensor and apply the view transformation
        tooth_n = torch.tensor(tooth_n, dtype=torch.long)
        marginline = torch.tensor(marginline, dtype=torch.float32).view(-1, 3)
        vertices = torch.tensor(vertices, dtype=torch.float32).view(-1, 3)
        data = Data(pos=vertices, y=marginline, tooth_n=tooth_n)
        return data
        # return vertices, marginline, tooth_n

# Usage of the dataset
def AtomicaMarginLine(args):
    # Create training and testing datasets
    train_dataset = MarginLineDataset(split='train', args = args)
    test_dataset = MarginLineDataset(split='test', args = args)

    # Create DataLoader for both
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader = PyGDataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    return train_loader, test_loader

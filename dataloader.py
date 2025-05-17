from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import fastmesh as fm
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
            tooth_id = case_id[-2:]
            tooth_enc = np.maximum(0, int(tooth_id) - 10 - 2 * ((int(tooth_id) // 10) - 1))

            bmesh_path = os.path.join(new_case_path, f"{tooth_id}_margin_context.bmesh")
            margin_path = os.path.join(new_case_path, f"{tooth_id}_margin.pts")

            data_list.append((bmesh_path, margin_path, tooth_enc))
        return data_list

    def __len__(self):
        return len(self.data_list)

    def _load_bmesh_file(self, bmesh_path):
        """Load .bemsh file, clean vertices using NumPy, and return processed vertices."""
        # Load the .bemsh file using trimesh
        vertices_np = fm.load(bmesh_path)[0]
        points = fps(vertices_np, self.centroids)[0]
        return points
 
    def _load_marginline(self, margin_path):
        """Load margin line from .pts file."""
        with open(margin_path, "r") as f:
            lines = f.readlines()
            marginline = np.array([list(map(float, line.split())) for line in lines[1:]])

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
        bmesh_path, margin_path, tooth_n = self.data_list[idx]

        marginline = self._load_marginline(margin_path)
        vertices = self._load_bmesh_file(bmesh_path)
        # Convert vertices to a PyTorch tensor and apply the view transformation
        tooth_n = torch.tensor(tooth_n, dtype=torch.long)
        marginline = torch.tensor(marginline, dtype=torch.float32).view(-1, 3)
        vertices = torch.tensor(vertices, dtype=torch.float32).view(-1, 3)
        return vertices, marginline, tooth_n

# Usage of the dataset
def AtomicaMarginLine(args):
    # Create training and testing datasets
    train_dataset = MarginLineDataset(split='train', args = args)
    test_dataset = MarginLineDataset(split='test', args = args)

    # Create DataLoader for both
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader

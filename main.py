import argparse
import os
from models.FoldingNet.FoldingNet import FoldingNet
from train import train
from dataloader import AtomicaMarginLine
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train FoldingNet on AtomicaMarginLine data.")

parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--output', type=str, default='./checkpoints', help='Output directory')
parser.add_argument('--model', type=str, default='my_model', help='Model name')
parser.add_argument('--centroids', type=int, default=2048, help='Number of centroids')
parser.add_argument('--marginNum', type=int, default=400, help='Number of margin points')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
parser.add_argument('--path', type=str, default="/home/waleed/Documents/3DLearning/margin-line/final/", help='base directory for data')

args = parser.parse_args()

# Optional: Create output dir if it doesn’t exist
os.makedirs(args.output, exist_ok=True)

train_loader, test_loader = AtomicaMarginLine(args)
model = FoldingNet(num_points=20).to('cuda')

train_res, test_res = train(model, train_loader, test_loader, args)

# ploting train and test
plt.figure()
plt.plot(train_res, label='Train Chamfer')
plt.plot(test_res, label='Test Chamfer')
plt.xlabel('Epochs')
plt.ylabel('Chamfer Loss')
plt.title('Training vs Test Chamfer')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(args.output, 'mIOU_plot.png'))
plt.close()
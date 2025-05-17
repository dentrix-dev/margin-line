import argparse
import os
from models.FoldingNet.FoldingNet import FoldingNet
from train import train
from dataloader import AtomicaMarginLine
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Create args
args = argparse.Namespace(
    lr=1e-3,
    num_epochs=100,
    output='./checkpoints',
    model='my_model',
    centroids= 2048,
    marginNum=400,
    batch_size=32,
    num_workers=4
)
# Optional: Create output dir if it doesn’t exist
os.makedirs(args.output, exist_ok=True)

train_loader, test_loader = AtomicaMarginLine(args.centroids, args.marginNum, args.batch_size, args.num_workers)
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
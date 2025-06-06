import os
import torch
from torch.amp import autocast
from tqdm import tqdm
from ChamferLoss import ChamferLoss
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter_mean

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

def center_data(data, batch_size=16, num_points=400):
    # Compute per-sample mean using the batch vector
    mean = scatter_mean(data.pos, data.batch, dim=0)
    batch_y = torch.arange(batch_size).repeat_interleave(num_points).to(device)
    data.pos -= mean[data.batch]
    data.y -= mean[batch_y]
    return data


def train(model, train_loader, test_loader, args):
    train_loss = []
    test_loss = []

    criterion = ChamferLoss()
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    writer = SummaryWriter(log_dir=args.output)
    best_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.train()
        cum_loss = 0

        for data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            data = center_data(data.to(device))

            with autocast(device_type='cuda' if cuda else 'cpu'):
                outputs = model(data)  # pass the full `Data` object

            loss = criterion(outputs.view(args.batch_size, args.marginNum, 3), data.y.view(args.batch_size, args.marginNum, 3))
            cum_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = cum_loss / len(train_loader)
        train_loss.append(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # Evaluation
        model.eval()
        t_loss = 0
        with torch.no_grad():
            for data in tqdm(test_loader, desc=f'[Eval] Epoch {epoch+1}/{args.num_epochs}'):
                data = center_data(data.to(device))

                with autocast(device_type='cuda' if cuda else 'cpu'):
                    outputs = model(data)

                t_loss += criterion(outputs.view(args.batch_size, args.marginNum, 3), data.y.view(args.batch_size, args.marginNum, 3)).item()

        avg_test_loss = t_loss / len(test_loader)
        test_loss.append(avg_test_loss)
        writer.add_scalar('Loss/test', avg_test_loss, epoch)

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model_path = os.path.join(args.output, f"{args.model}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"[✓] Best model saved with test loss: {best_loss:.4f}")

        scheduler.step()
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], train_Loss: {avg_train_loss:.4f}, test_Loss: {avg_test_loss:.4f}')

    writer.close()
    print('Training finished.')
    return train_loss, test_loss

import os
import torch
from torch.amp import autocast
from tqdm import tqdm
from ChamferLoss import ChamferLoss
from torch.utils.tensorboard import SummaryWriter

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'

def train(model, train_loader, test_loader, args):
    train_loss = []
    test_loss = []

    criterion = ChamferLoss()
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # TensorBoard setup
    writer = SummaryWriter(log_dir=args.output)

    best_loss = float('inf')  # For saving best model

    for epoch in range(args.num_epochs):
        model.train()
        cum_loss = 0

        for vertices, margin_line, teeth in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            vertices, margin_line, teeth = vertices.to(device), margin_line.to(device), teeth.to(device)
            mean = vertices.mean(dim=1, keepdim=True)
            vertices -= mean
            margin_line -= mean

            with autocast(device_type='cuda'):
                outputs = model(vertices, teeth)

            loss = criterion(outputs, margin_line)
            cum_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Average training loss
        cum_loss /= len(train_loader)
        train_loss.append(cum_loss)
        writer.add_scalar('Loss/train', cum_loss, epoch)

        # Evaluation
        model.eval()
        t_loss = 0
        with torch.no_grad():
            for vertices, margin_line, teeth in tqdm(test_loader, desc=f'[Eval] Epoch {epoch+1}/{args.num_epochs}'):
                vertices, margin_line, teeth = vertices.to(device), margin_line.to(device), teeth.to(device)
                mean = vertices.mean(dim=1, keepdim=True)
                vertices -= mean
                margin_line -= mean

                with autocast(device_type='cuda'):
                    outputs = model(vertices, teeth)

                t_loss += criterion(outputs, margin_line).item()

        # Average test loss
        t_loss /= len(test_loader)
        test_loss.append(t_loss)
        writer.add_scalar('Loss/test', t_loss, epoch)

        # Save best model
        if t_loss < best_loss:
            best_loss = t_loss
            best_model_path = os.path.join(args.output, f"{args.model}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"[✓] Best model saved with test loss: {best_loss:.4f}")

        # Step the scheduler
        scheduler.step()

        print(f'Epoch [{epoch + 1}/{args.num_epochs}], train_Loss: {cum_loss:.4f}, test_Loss: {t_loss:.4f}')

    writer.close()
    print('Training finished.')

    return train_loss, test_loss
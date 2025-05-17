from dataloader import AtomicaMarginLine

train_loader, test_loader = AtomicaMarginLine(1024, 400, 32, 8)

for vertices, marginline, teeth in train_loader:
    print(vertices.shape, marginline.shape, teeth.shape)

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, scheduler, logger, args):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % args.log_interval == 0:
            logger.info(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} '
                f'({100.0 * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def train(model, criterion, data_loader, optimizer, device, scheduler, logger, args):
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, scheduler, logger, args)
        scheduler.step()

# Example usage
model = ConvNeXt_XL(in_channels=3, out_channels=96, drop_path_rate=0.5, layer_scale_init_value=1e-6)
criterion = nn.CrossEntropyLoss()
data_loader = torch.utils.data.DataLoader(datasets.ImageNet(args.data_path, split='train', transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])), batch_size=256, shuffle=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
device = torch.device('cuda')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
logger = Logger(args.log_file)
train(model, criterion, data_loader, optimizer, device, scheduler, logger, args)
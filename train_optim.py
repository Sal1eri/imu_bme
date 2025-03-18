def load_model(args):
    """
    Load model based on arguments
    Args:
        args: Command line arguments
    Returns:
        model_name: Name of the model
        net: Loaded model
    """
    model_map = {
        'Unet': ('UNet', UNet(3, 2)),
        'FCN': ('FCN8x', FCN8x(args.n_classes)),
        'Deeplab': ('DeepLabV3', DeepLabV3(args.n_classes)),
        'Unet3+': ('Unet3+', UNet_3Plus()),
        'Qnet': ('Qnet', ResNetUNet()),
        'Uesnet50': ('Uesnet50', UesNet()),
        'Unet2+': ('URestnet++', NestedUResnet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=2)),
        'ViT': ('ViT', ViT_UNet(num_classes=2)),
        'PSPnet': ('PSPnet', PSPNet(3))
    }

    model_name, net = model_map.get(args.model, model_map['PSPnet'])
    print(f"Using {model_name}")

    return model_name, net


def setup_training(args, model_name):
    """
    Setup training environment and directories
    Args:
        args: Command line arguments
        model_name: Name of the model
    Returns:
        model_path: Path to save model
        result_path: Path to save results
        plots_dir: Directory for plots
        timestamp: Current timestamp
    """
    # Setup paths
    model_path = f'./model_result/best_model_{model_name}.mdl'
    result_path = f'./result_{model_name}.txt'
    plots_dir = os.path.join('training_plots', model_name)

    # Create directories
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Remove existing result file
    if os.path.exists(result_path):
        os.remove(result_path)

    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    return model_path, result_path, plots_dir, timestamp


def setup_data_loaders(args):
    """
    Setup data loaders with transforms
    Args:
        args: Command line arguments
    Returns:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
    """
    # Define transforms
    train_transform = Compose([
        Resize((args.input_height, args.input_width)),
        RandomHorizontalFlip(0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = Compose([
        Resize((args.input_height, args.input_width)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = SegData(
        image_path=os.path.join(args.data_path, 'training/images'),
        mask_path=os.path.join(args.data_path, 'training/segmentations'),
        data_transforms=train_transform
    )

    val_dataset = SegData(
        image_path=os.path.join(args.data_path, 'validation/images'),
        mask_path=os.path.join(args.data_path, 'validation/segmentations'),
        data_transforms=val_transform
    )

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader


def train(args, model_name, net):
    """
    Main training function
    Args:
        args: Command line arguments
        model_name: Name of the model
        net: Model to train
    """
    # Setup training environment
    model_path, result_path, plots_dir, timestamp = setup_training(args, model_name)

    # Setup data loaders
    train_dataloader, val_dataloader = setup_data_loaders(args)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        net = net.cuda()

    # Setup loss functions and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    surface_criterion = SurfaceLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.init_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    # Training parameters
    alpha, beta = 0, 1  # Loss weights
    best_score = 0.0
    start_time = time.time()

    # Initialize history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_acc_cls': [], 'val_acc_cls': [],
        'train_mean_iu': [], 'val_mean_iu': [],
        'train_fwavacc': [], 'val_fwavacc': [],
        'learning_rates': []
    }

    # Training loop
    for e in range(args.epochs):
        # Training phase
        train_metrics = train_epoch(net, train_dataloader, criterion, surface_criterion,
                                    optimizer, device, alpha, beta, e, args.epochs)

        # Validation phase
        val_metrics = validate(net, val_dataloader, criterion, surface_criterion,
                               device, alpha, beta, e, args.epochs)

        # Update learning rate
        scheduler.step(train_metrics['loss'])

        # Update history
        update_history(history, train_metrics, val_metrics, optimizer.param_groups[0]['lr'])

        # Save results
        save_results(result_path, e, train_metrics, val_metrics)

        # Plot and save metrics
        plot_training_curves(history, plots_dir)
        save_history(history, plots_dir, timestamp)

        # Save best model
        score = (val_metrics['acc_cls'] + val_metrics['mean_iu']) / 2
        if score > best_score:
            best_score = score
            torch.save(net.state_dict(), model_path)
            print(f'Best model saved with score: {best_score:.4f}')

    print(f'Total training time: {time.time() - start_time:.2f} seconds')


def train_epoch(net, dataloader, criterion, surface_criterion, optimizer, device, alpha, beta, epoch, total_epochs):
    """
    Train for one epoch
    """
    net.train()
    epoch_start_time = time.time()
    train_loss = 0.0
    label_true = torch.LongTensor()
    label_pred = torch.LongTensor()

    with tqdm(total=len(dataloader), desc=f'{epoch + 1}/{total_epochs} epoch Train_Progress') as pb:
        for batchdata, batchlabel in dataloader:
            batchdata = batchdata.to(device)
            batchlabel = (batchlabel / 255).to(device).long()

            # Forward pass
            output = net(batchdata)
            output = F.log_softmax(output, dim=1)

            # Calculate loss
            ce_loss = criterion(output, batchlabel)
            bd_loss = boundary_loss(output, batchlabel)
            loss = alpha * ce_loss + beta * bd_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            # Record metrics
            train_loss += loss.item() * batchlabel.size(0)
            pred = output.argmax(dim=1).squeeze().data.cpu()
            real = batchlabel.data.cpu()
            label_true = torch.cat((label_true, real), dim=0)
            label_pred = torch.cat((label_pred, pred), dim=0)

            pb.update(1)

    # Calculate metrics
    train_loss /= len(dataloader.dataset)
    acc, acc_cls, mean_iu, fwavacc, _, _, _, _ = label_accuracy_score(
        label_true.numpy(), label_pred.numpy(), 2
    )

    print(f'epoch: {epoch + 1}')
    print(f'train_loss: {train_loss:.4f}, acc: {acc:.4f}, acc_cls: {acc_cls:.4f}, '
          f'mean_iu: {mean_iu:.4f}, fwavacc: {fwavacc:.4f}')
    print(f'Time for this epoch: {time.time() - epoch_start_time:.2f} seconds')

    return {
        'loss': train_loss,
        'acc': acc,
        'acc_cls': acc_cls,
        'mean_iu': mean_iu,
        'fwavacc': fwavacc
    }


def validate(net, dataloader, criterion, surface_criterion, device, alpha, beta, epoch, total_epochs):
    """
    Validate the model
    """
    net.eval()
    val_loss = 0.0
    val_label_true = torch.LongTensor()
    val_label_pred = torch.LongTensor()

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc=f'{epoch + 1}/{total_epochs} epoch Val_Progress') as pb:
            for batchdata, batchlabel in dataloader:
                batchdata = batchdata.to(device)
                batchlabel = (batchlabel / 255).to(device).long()

                output = net(batchdata)
                output = F.log_softmax(output, dim=1)

                ce_loss = criterion(output, batchlabel)
                bd_loss = boundary_loss(output, batchlabel)
                loss = alpha * ce_loss + beta * bd_loss

                pred = output.argmax(dim=1).squeeze().data.cpu()
                real = batchlabel.data.cpu()

                val_loss += loss.item() * batchlabel.size(0)
                val_label_true = torch.cat((val_label_true, real), dim=0)
                val_label_pred = torch.cat((val_label_pred, pred), dim=0)

                pb.update(1)

    val_loss /= len(dataloader.dataset)
    val_acc, val_acc_cls, val_mean_iu, val_fwavacc, _, _, _, _ = label_accuracy_score(
        val_label_true.numpy(), val_label_pred.numpy(), 2
    )

    print(f'val_loss: {val_loss:.4f}, acc: {val_acc:.4f}, acc_cls: {val_acc_cls:.4f}, '
          f'mean_iu: {val_mean_iu:.4f}, fwavacc: {val_fwavacc:.4f}')

    return {
        'loss': val_loss,
        'acc': val_acc,
        'acc_cls': val_acc_cls,
        'mean_iu': val_mean_iu,
        'fwavacc': val_fwavacc
    }


def update_history(history, train_metrics, val_metrics, lr):
    """
    Update training history
    """
    history['train_loss'].append(train_metrics['loss'])
    history['train_acc'].append(train_metrics['acc'])
    history['train_acc_cls'].append(train_metrics['acc_cls'])
    history['train_mean_iu'].append(train_metrics['mean_iu'])
    history['train_fwavacc'].append(train_metrics['fwavacc'])

    history['val_loss'].append(val_metrics['loss'])
    history['val_acc'].append(val_metrics['acc'])
    history['val_acc_cls'].append(val_metrics['acc_cls'])
    history['val_mean_iu'].append(val_metrics['mean_iu'])
    history['val_fwavacc'].append(val_metrics['fwavacc'])

    history['learning_rates'].append(lr)


def save_results(result_path, epoch, train_metrics, val_metrics):
    """
    Save training results to file
    """
    with open(result_path, 'a') as f:
        f.write(f'\nepoch: {epoch + 1}\n')
        f.write(f'train_loss: {train_metrics["loss"]:.4f}, acc: {train_metrics["acc"]:.4f}, '
                f'acc_cls: {train_metrics["acc_cls"]:.4f}, mean_iu: {train_metrics["mean_iu"]:.4f}, '
                f'fwavacc: {train_metrics["fwavacc"]:.4f}\n')


def save_history(history, plots_dir, timestamp):
    """
    Save training history to CSV
    """
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(plots_dir, f'training_history_{timestamp}.csv'), index=False)
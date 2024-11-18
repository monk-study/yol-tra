def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    batch_count = 0
    grad_norms = []
    branch_correct = {branch: 0 for branch in model.module.nba_branches.keys()}
    branch_total = {branch: 0 for branch in model.module.nba_branches.keys()}
    
    # Ensure model is on CPU
    model = model.cpu()
    
    for features, labels in train_loader:
        # Explicitly move everything to CPU
        features = features.cpu()
        labels = labels.cpu()
        
        optimizer.zero_grad()
        outputs = model(features)
        
        # Create one-hot labels on CPU
        labels_onehot = F.one_hot(labels, num_classes=len(model.module.nba_branches)).float().cpu()
        
        # Ensure criterion is on CPU with CPU weights
        pos_weight = torch.ones([len(model.module.nba_branches)]).cpu() * 2.0
        batch_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        loss = batch_criterion(outputs, labels_onehot)
        loss.backward()
        
        # Rest of the function remains the same...

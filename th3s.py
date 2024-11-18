def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    batch_count = 0
    grad_norms = []
    branch_correct = {branch: 0 for branch in model.module.nba_branches.keys()}
    branch_total = {branch: 0 for branch in model.module.nba_branches.keys()}
    
    for features, labels in train_loader:
        # Debug device placement
        print(f"\nBatch {batch_count} device check:")
        print(f"Features device: {features.device}")
        print(f"Labels device: {labels.device}")
        
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        print(f"Outputs device: {outputs.device}")
        
        labels_onehot = F.one_hot(labels, num_classes=len(model.module.nba_branches)).float()
        labels_onehot = labels_onehot.to(device)
        print(f"Labels one-hot device: {labels_onehot.device}")
        
        # Print criterion device info
        for param in criterion.parameters():
            print(f"Criterion param device: {param.device}")
            
        # Create new criterion for this batch to ensure device matching
        pos_weight = torch.ones([len(model.module.nba_branches)]).to(outputs.device) * 2.0
        batch_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        try:
            loss = batch_criterion(outputs, labels_onehot)
        except Exception as e:
            print("Error in loss calculation:")
            print(f"outputs shape: {outputs.shape}, device: {outputs.device}")
            print(f"labels_onehot shape: {labels_onehot.shape}, device: {labels_onehot.device}")
            print(f"pos_weight shape: {pos_weight.shape}, device: {pos_weight.device}")
            raise e

        loss.backward()
        
        # Rest of your function remains the same...

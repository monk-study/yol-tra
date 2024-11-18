def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    batch_count = 0
    
    # Initialize gradient scaler
    scaler = torch.cuda.amp.GradScaler()
    
    for features, labels in train_loader:
        features = features.cpu()
        labels = labels.cpu()
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)  # More efficient
        
        # Forward pass with gradient scaling
        with torch.autocast(device_type='cpu', dtype=torch.float32):
            outputs = model(features)
            labels_onehot = F.one_hot(labels, num_classes=len(model.module.nba_branches)).float()
            loss = criterion(outputs, labels_onehot)
        
        # Backward pass with gradient scaling
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()  # Step per batch
        
        total_loss += loss.item()
        
        # Print more frequent updates
        if batch_count % 100 == 0:
            print(f"\rBatch {batch_count}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        batch_count += 1

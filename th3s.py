config = {
    'shared_dims': [256, 128],  # Reduced model size
    'branch_dims': [64],        # Simplified branches
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 128,          # Increased batch size
    'epochs': 50,
    'early_stopping_patience': 5,
    'weight_decay': 0.01
}
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    batch_count = 0
    grad_norms = []  # Track gradient norms
    branch_correct = {branch: 0 for branch in model.module.nba_branches.keys()}
    branch_total = {branch: 0 for branch in model.module.nba_branches.keys()}
    
    # For quick validation during training
    early_stopping_batches = len(train_loader) // 4  # Check every 25% of epoch
    
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        
        # Monitor output distributions
        if batch_count % 100 == 0:
            print(f"\nBatch {batch_count} stats:")
            print(f"Outputs mean: {outputs.mean():.4f}, std: {outputs.std():.4f}")
            print(f"Outputs range: [{outputs.min():.4f}, {outputs.max():.4f}]")
        
        labels_onehot = F.one_hot(labels, num_classes=len(model.module.nba_branches)).float()
        labels_onehot = labels_onehot.to(outputs.device)
        
        # Using BCEWithLogitsLoss with class weights
        pos_weight = torch.ones([len(model.module.nba_branches)], device=device) * 2.0  # Adjust this weight
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(outputs, labels_onehot)
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Monitor gradients
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        predictions = (outputs > 0).float()
        
        # Calculate accuracies per branch
        for i, branch_name in enumerate(model.module.nba_branches.keys()):
            branch_correct[branch_name] += (predictions[:, i] == labels_onehot[:, i]).sum().item()
            branch_total[branch_name] += labels_onehot[:, i].sum().item()
        
        # Quick validation check every 25% of epoch
        if batch_count % early_stopping_batches == 0 and batch_count > 0:
            quick_metrics = quick_validate(model, train_loader, criterion, device, num_batches=50)
            print(f"\nQuick validation at batch {batch_count}:")
            print(f"Current loss: {quick_metrics['loss']:.4f}")
            print(f"Current accuracy: {quick_metrics['accuracy']:.4f}")
            
            # Early stopping check
            if quick_metrics['loss'] < best_loss:
                best_loss = quick_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 3:  # Adjust patience as needed
                    print("Early stopping triggered during epoch")
                    break
        
        batch_count += 1
        
        # Print progressive training stats
        if batch_count % 10 == 0:  # Adjust frequency as needed
            current_loss = total_loss / (batch_count + 1)
            print(f"\rBatch {batch_count}/{len(train_loader)}, "
                  f"Loss: {current_loss:.4f}, "
                  f"Grad norm: {total_norm:.4f}", end="")
    
    avg_loss = total_loss / len(train_loader)
    
    # Calculate metrics
    branch_accuracies = {
        branch: correct/total if total > 0 else 0 
        for branch, (correct, total) in zip(
            branch_correct.keys(),
            zip(branch_correct.values(), branch_total.values())
        )
    }
    
    total_correct = sum(branch_correct.values())
    total_samples = sum(branch_total.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # Print training statistics
    print("\n\nEpoch Summary:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Average Gradient Norm: {sum(grad_norms)/len(grad_norms):.4f}")
    print("\nBranch Accuracies:")
    for branch, acc in branch_accuracies.items():
        print(f"{branch}: {acc:.4f}")
    
    return avg_loss, overall_accuracy, branch_accuracies

def quick_validate(model, dataloader, criterion, device, num_batches=50):
    """Quick validation on a small subset of data"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    
    with torch.no_grad():
        for features, labels in dataloader:
            if batch_count >= num_batches:
                break
                
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            labels_onehot = F.one_hot(labels, num_classes=len(model.module.nba_branches)).float()
            labels_onehot = labels_onehot.to(outputs.device)
            
            loss = criterion(outputs, labels_onehot)
            predictions = (outputs > 0).float()
            
            total_loss += loss.item()
            correct += (predictions == labels_onehot).all(dim=1).sum().item()
            total += labels.size(0)
            batch_count += 1
    
    return {
        'loss': total_loss / batch_count,
        'accuracy': correct / total if total > 0 else 0
    }

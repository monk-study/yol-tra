multibranch_configs = {
    'shallow': {
        'shared_dims': [256],         # Reduced from [256, 128]
        'branch_dims': [64],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 512,            # Increased from 128
        'epochs': 50,
        'early_stopping_patience': 5,
        'weight_decay': 0.01
    }
}

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    batch_count = 0
    grad_norms = []
    
    for features, labels in train_loader:
        features = features.cpu()
        labels = labels.cpu()
        
        optimizer.zero_grad(set_to_none=True)  # More efficient
        
        outputs = model(features)
        labels_onehot = F.one_hot(labels, num_classes=len(model.nba_branches)).float()
        
        loss = criterion(outputs, labels_onehot)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Simplified gradient monitoring
        if batch_count % 50 == 0:  # Reduced monitoring frequency
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        correct, samples = get_correct_prediction(outputs, labels_onehot)
        total_correct += correct
        total_samples += samples
        
        # Print less frequently
        if batch_count % 50 == 0:  # Changed from 10
            current_loss = total_loss / (batch_count + 1)
            current_acc = (total_correct / total_samples) if total_samples > 0 else 0
            print(f"\rBatch {batch_count}/{len(train_loader)}, "
                  f"Loss: {current_loss:.4f}, "
                  f"Acc: {current_acc:.4f}", end="")
        
        batch_count += 1
    
    avg_loss = total_loss / len(train_loader)
    accuracy = (total_correct / total_samples) if total_samples > 0 else 0
    
    print("\n\nEpoch Summary:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    if grad_norms:
        print(f"Average Gradient Norm: {sum(grad_norms)/len(grad_norms):.4f}")
    
    return avg_loss, accuracy

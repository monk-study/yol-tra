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
        
        optimizer.zero_grad(set_to_none=True)
        
        outputs = model(features)
        # Change here: model.nba_branches instead of model.module.nba_branches
        labels_onehot = F.one_hot(labels, num_classes=len(model.nba_branches)).float()
        
        loss = criterion(outputs, labels_onehot)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
        
        if batch_count % 10 == 0:
            current_loss = total_loss / (batch_count + 1)
            current_acc = (total_correct / total_samples) if total_samples > 0 else 0
            print(f"\rBatch {batch_count}/{len(train_loader)}, "
                  f"Loss: {current_loss:.4f}, "
                  f"Acc: {current_acc:.4f}, "
                  f"Grad norm: {total_norm:.4f}", end="")
        
        batch_count += 1
    
    avg_loss = total_loss / len(train_loader)
    accuracy = (total_correct / total_samples) if total_samples > 0 else 0
    
    print("\n\nEpoch Summary:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Gradient Norm: {sum(grad_norms)/len(grad_norms):.4f}")
    
    return avg_loss, accuracy

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.cpu()
            labels = labels.cpu()
            
            outputs = model(features)
            # Change here: model.nba_branches instead of model.module.nba_branches
            labels_onehot = F.one_hot(labels, num_classes=len(model.nba_branches)).float()
            
            loss = criterion(outputs, labels_onehot)
            total_loss += loss.item()
            
            correct, samples = get_correct_prediction(outputs, labels_onehot)
            total_correct += correct
            total_samples += samples
            
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels_onehot.cpu().numpy())
    
    accuracy = (total_correct / total_samples) if total_samples > 0 else 0
    
    all_outputs = np.vstack(all_outputs)
    all_labels = np.vstack(all_labels)
    predictions = (all_outputs > 0).astype(float)
    
    metrics = {}
    # Change here: model.nba_branches instead of model.module.nba_branches
    for i, branch_name in enumerate(model.nba_branches.keys()):
        metrics[branch_name] = {
            'precision': precision_score(all_labels[:, i], predictions[:, i], zero_division=0),
            'recall': recall_score(all_labels[:, i], predictions[:, i], zero_division=0),
            'f1': f1_score(all_labels[:, i], predictions[:, i], zero_division=0),
            'support': all_labels[:, i].sum()
        }
    
    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'branch_metrics': pd.DataFrame(metrics).T,
        'predictions': predictions,
        'labels': all_labels
    }

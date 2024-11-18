def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    batch_count = 0
    grad_norms = []
    branch_correct = {branch: 0 for branch in model.module.nba_branches.keys()}
    branch_total = {branch: 0 for branch in model.module.nba_branches.keys()}
    
    for features, labels in train_loader:
        features = features.cpu()
        labels = labels.cpu()
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)  # More efficient
        
        # Forward pass
        outputs = model(features)
        labels_onehot = F.one_hot(labels, num_classes=len(model.module.nba_branches)).float()
        loss = criterion(outputs, labels_onehot)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
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
        
        if batch_count % 10 == 0:
            current_loss = total_loss / (batch_count + 1)
            print(f"\rBatch {batch_count}/{len(train_loader)}, "
                  f"Loss: {current_loss:.4f}, "
                  f"Grad norm: {total_norm:.4f}", end="")
        
        batch_count += 1
    
    # Calculate final metrics
    avg_loss = total_loss / len(train_loader)
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
    
    # Print epoch summary
    print("\n\nEpoch Summary:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Average Gradient Norm: {sum(grad_norms)/len(grad_norms):.4f}")
    print("\nBranch Accuracies:")
    for branch, acc in branch_accuracies.items():
        print(f"{branch}: {acc:.4f}")
    
    return avg_loss, overall_accuracy, branch_accuracies

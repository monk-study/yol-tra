def execute_multibranch_model(params, X_columns, y_column, config_name):
    device = torch.device('cpu')
    torch.set_num_threads(4)
    
    # Initialize history dict with all required keys
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'branch_metrics': []
    }
    
    # ... (rest of initialization code)
    
    for epoch in range(params['epochs']):
        # Training phase
        train_loss, train_acc, train_branch_accs = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation phase
        eval_results = evaluate(model, valid_loader, criterion, device)
        val_loss = eval_results['loss']
        # Calculate validation accuracy from branch metrics
        val_acc = eval_results['branch_metrics']['accuracy'].mean()
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['branch_metrics'].append(eval_results['branch_metrics'])
        
        print(f'\nEpoch {epoch+1}/{params["epochs"]}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print('\nBranch Metrics:')
        print(eval_results['branch_metrics'])
        
        # Rest of the code...
    
    return model, history

#----------
def get_branch_metrics(predictions, probabilities, labels):
    metrics = {}
    for branch in predictions.keys():
        y_true = np.array(labels[branch])
        y_pred = np.array(predictions[branch])
        y_prob = np.array(probabilities[branch])
        
        metrics[branch] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob)
        }
    
    return pd.DataFrame(metrics).T
#----------
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    batch_count = 0
    
    # ... (rest of the code)
    
    for features, labels in train_loader:
        # Add debugging prints every 100 batches
        if batch_count % 100 == 0:
            print(f"\nBatch {batch_count}")
            print(f"Features range: {features.min().item():.3f} to {features.max().item():.3f}")
            print(f"Unique labels: {torch.unique(labels).numpy()}")
        
        # ... (rest of the batch processing)
        
        batch_count += 1

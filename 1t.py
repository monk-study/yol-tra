def execute_nn_baseline(params, X_columns, y_column, config_name):
    print('\nStarting training')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Label encoding
    global le
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(train_df[y_column])
    y_valid_encoded = le.transform(valid_df[y_column])
    y_test_encoded = le.transform(test_df[y_column])
    
    # Create datasets and loaders
    train_dataset = TPRRDataset(train_df[X_columns].values, y_train_encoded)
    valid_dataset = TPRRDataset(valid_df[X_columns].values, y_valid_encoded)
    test_dataset = TPRRDataset(test_df[X_columns].values, y_test_encoded)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])
    
    model = TPRRClassifier(
        input_dim=len(X_columns),
        hidden_dims=params['hidden_dims'],
        num_classes=len(le.classes_),
        dropout_rate=params['dropout_rate']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience = params['early_stopping_patience']
    patience_counter = 0
    best_model_path = f'best_model_{config_name}.pt'
    
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [],
        'train_precision': [], 'train_recall': [], 'train_f1': [], 'train_auc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auc': []
    }
    
    def calculate_metrics(labels, predictions, probabilities):
        metrics = {}
        metrics['acc'] = accuracy_score(labels, predictions)
        
        # Calculate per-class metrics and average them
        precisions = []
        recalls = []
        f1s = []
        aucs = []
        
        for class_idx in range(len(le.classes_)):
            y_true_binary = (labels == class_idx)
            y_pred_binary = (predictions == class_idx)
            y_prob_binary = probabilities[:, class_idx]
            
            precisions.append(precision_score(y_true_binary, y_pred_binary))
            recalls.append(recall_score(y_true_binary, y_pred_binary))
            f1s.append(f1_score(y_true_binary, y_pred_binary))
            aucs.append(roc_auc_score(y_true_binary, y_prob_binary))
        
        metrics['precision'] = np.mean(precisions)
        metrics['recall'] = np.mean(recalls)
        metrics['f1'] = np.mean(f1s)
        metrics['auc'] = np.mean(aucs)
        
        return metrics
    
    print("\nStarting training...")
    for epoch in range(params['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_probabilities = []
        train_labels = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            probs = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            train_predictions.extend(predicted.cpu().numpy())
            train_probabilities.extend(probs.cpu().numpy())
            train_labels.extend(target.cpu().numpy())
        
        train_loss = train_loss / len(train_loader)
        train_metrics = calculate_metrics(
            np.array(train_labels), 
            np.array(train_predictions),
            np.array(train_probabilities)
        )
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_probabilities = []
        val_labels = []
        
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                probs = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                val_predictions.extend(predicted.cpu().numpy())
                val_probabilities.extend(probs.cpu().numpy())
                val_labels.extend(target.cpu().numpy())
        
        val_loss = val_loss / len(valid_loader)
        val_metrics = calculate_metrics(
            np.array(val_labels), 
            np.array(val_predictions),
            np.array(val_probabilities)
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_metrics['acc'])
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_auc'].append(train_metrics['auc'])
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['acc'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        print(f'\nEpoch {epoch+1}/{params["epochs"]}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_metrics["acc"]:.4f}')
        print(f'Train Precision: {train_metrics["precision"]:.4f}, Train Recall: {train_metrics["recall"]:.4f}')
        print(f'Train F1: {train_metrics["f1"]:.4f}, Train AUC: {train_metrics["auc"]:.4f}')
        print(f'Valid Loss: {val_loss:.4f}, Valid Acc: {val_metrics["acc"]:.4f}')
        print(f'Valid Precision: {val_metrics["precision"]:.4f}, Valid Recall: {val_metrics["recall"]:.4f}')
        print(f'Valid F1: {val_metrics["f1"]:.4f}, Valid AUC: {val_metrics["auc"]:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_columns': X_columns,
                'label_encoder_classes': le.classes_,
                'history': history
            }, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered')
                break
    
    # Load best model for final evaluation
    saved_data = torch.load(best_model_path)
    model.load_state_dict(saved_data['model_state_dict'])
    
    return model, history

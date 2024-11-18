def execute_multibranch_model(params, X_columns, y_column, config_name):
    # Force CPU usage and set threads
    device = torch.device('cpu')
    torch.set_num_threads(6)  # Using your 6 CPU cores
    print(f'Using device: {device}')
    print(f'Using {torch.get_num_threads()} CPU threads')
    
    # Label encoding setup
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(train_df[y_column])
    y_valid_encoded = le.transform(valid_df[y_column])
    y_test_encoded = le.transform(test_df[y_column])
    
    # Create datasets
    train_dataset = TPRRDataset(train_df[X_columns].values, y_train_encoded)
    valid_dataset = TPRRDataset(valid_df[X_columns].values, y_valid_encoded)
    test_dataset = TPRRDataset(test_df[X_columns].values, y_test_encoded)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])
    
    # Initialize model and ensure it's on CPU
    model = TPRRMultiBranchClassifier(
        input_dim=len(X_columns),
        shared_dims=params['shared_dims'],
        branch_dims=params['branch_dims'],
        dropout_rate=params['dropout_rate']
    ).to(device)
    
    # Create criterion with pos_weight on same device
    pos_weight = torch.ones([len(model.nba_branches)]).to(device) * 2.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    
    # Initialize optimizer after model is on correct device
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    
    # Initialize scheduler
    scheduler = torch.optim.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'branch_metrics': []
    }
    
    # Training loop setup
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(params['epochs']):
        # Training phase
        train_loss, train_acc, train_branch_accs = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation phase
        eval_results = evaluate(model, valid_loader, criterion, device)
        val_loss = eval_results['loss']
        val_acc = eval_results['branch_metrics']['accuracy'].mean()
        
        # Update learning rate
        scheduler.step(val_loss)
        
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
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': params,
                'history': history
            }, f'best_model_multibranch_{config_name}.pt')
            print(f"New best model saved! Validation Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= params['early_stopping_patience']:
                print('Early stopping triggered')
                break
    
    return model, history

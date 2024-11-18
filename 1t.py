def execute_multibranch_model(params, X_columns, y_column, config_name):
    # ... (previous code until training loop)
    
    print("\nStarting training...")
    for epoch in range(params['epochs']):
        # Training phase
        train_loss, train_acc, train_branch_accs = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,  # Added scheduler
            device=device
        )
        
        # Validation phase
        eval_results = evaluate(model, valid_loader, criterion, device)
        
        # ... (rest of the code)

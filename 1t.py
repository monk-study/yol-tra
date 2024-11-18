def execute_multibranch_model(params, X_columns, y_column, config_name):
    # ... (previous code)
    
    # Calculate class weights based on data distribution
    class_counts = train_df[y_column].value_counts()
    total_samples = len(train_df)
    class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) 
                                     for count in class_counts]).to(device)
    
    # Initialize model with different architecture
    model = TPRRMultiBranchClassifier(
        input_dim=len(X_columns),
        shared_dims=[512, 256],  # Larger shared network
        branch_dims=[128, 64],   # Deeper branch networks
        dropout_rate=0.3
    ).to(device)
    
    # Use weighted loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # Use different optimizer settings
    optimizer = torch.optim.AdamW(  # Switch to AdamW
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Add learning rate scheduling
    scheduler = torch.optim.OneCycleLR(  # Use OneCycleLR instead
        optimizer,
        max_lr=0.001,
        epochs=params['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

def execute_multibranch_model(params, X_columns, y_column, config_name):
    # Force CPU
    torch.cuda.is_available = lambda : False  # Force CPU usage
    device = torch.device('cpu')
    torch.set_num_threads(6)
    print(f'Using device: {device}')
    print(f'Using {torch.get_num_threads()} CPU threads')
    
    # Create datasets without device specification
    train_dataset = TPRRDataset(train_df[X_columns].values, y_train_encoded)
    valid_dataset = TPRRDataset(valid_df[X_columns].values, y_valid_encoded)
    test_dataset = TPRRDataset(test_df[X_columns].values, y_test_encoded)
    
    # Rest of initialization...

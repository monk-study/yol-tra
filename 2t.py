# First load the model
model_path = "nn_ft_prob.pt"
saved_data = torch.load(model_path, map_location=torch.device('cpu'))

# Create test dataset
test_dataset = TPRRDataset(
    test_df[X_columns].values,
    le.transform(test_df[y_column])
)

# Create test loader
test_loader = DataLoader(
    test_dataset,
    batch_size=32,  # you can adjust batch size
    shuffle=False   # don't shuffle for evaluation
)

# Now run the evaluation
results = evaluate_saved_model(model_path, test_loader, device='cpu')  # specify device='cpu' if not using CUDA

# Rest of your code for printing metrics and plotting

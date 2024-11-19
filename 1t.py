# Create test dataset using the same features as training
model_path = "nn_ft_prob.pt"
saved_data = torch.load(model_path, map_location='cpu')
features_to_use = saved_data['feature_columns']

# Create test dataset
test_dataset = TPRRDataset(
    test_df[features_to_use].values,
    le.transform(test_df[y_column])
)

# Create test loader
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Run evaluation
results, saved_data = evaluate_saved_model(model_path, test_loader, device='cpu')

# Print metrics and create plots as before
print("\nOverall Accuracy:", results['accuracy'])

print("\nPer-class Metrics:")
for class_name, metrics in results['per_class_metrics'].items():
    print(f"\n{class_name}:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"AUC-ROC: {metrics['auc']:.4f}")

# Analyze predictions
analysis = analyze_predictions(results)

# Plot confusion matrix
plot_confusion_matrix(
    results['true_labels'],
    results['predictions'],
    classes=saved_data['label_encoder_classes'],
    normalize=True,
    title='Normalized Confusion Matrix'
)

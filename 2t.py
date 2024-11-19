# Cell 4: Usage example
model_path = "nn_ft_prob.pt"
results = evaluate_saved_model(model_path, test_loader)

# Print comprehensive metrics
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

# Plot confusion matrix using existing function
plot_confusion_matrix(
    results['true_labels'],
    results['predictions'],
    classes=saved_data['label_encoder_classes'],
    normalize=True,
    title='Normalized Confusion Matrix'
)

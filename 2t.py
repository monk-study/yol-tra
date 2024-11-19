# Load best model and evaluate
model_path = "nn_ft_prob.pt"
saved_data = torch.load(model_path)
model.load_state_dict(saved_data['model_state_dict'])
model.eval()

# Get predictions and probabilities
test_preds = []
test_probs = []
test_labels = []

with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        test_preds.extend(predicted.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
test_preds = np.array(test_preds)
test_probs = np.array(test_probs)
test_labels = np.array(test_labels)

# Use existing function for confusion matrix
fig, ax = plot_confusion_matrix(
    test_labels,
    test_preds,
    classes=le.classes_,
    normalize=True,
    title='Neural Network Confusion Matrix'
)

# Print standard classification report
print("\nClassification Report:")
print(classification_report(test_labels, test_preds))

# Calculate and print per-class metrics
print("\nDetailed Per-Class Metrics:")
for idx, class_name in enumerate(le.classes_):
    y_true_binary = (test_labels == idx)
    y_pred_binary = (test_preds == idx)
    
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    auc = roc_auc_score(y_true_binary, test_probs[:, idx])
    
    print(f"\n{class_name}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

# Analyze prediction confidence
results = {
    'probabilities': test_probs,
    'predictions': test_preds,
    'true_labels': test_labels
}

confidence_analysis = analyze_predictions(results)
print("\nConfidence Analysis:")
print(f"High confidence correct predictions: {confidence_analysis['high_conf_correct']}")
print(f"Low confidence correct predictions: {confidence_analysis['low_conf_correct']}")
print(f"High confidence incorrect predictions: {confidence_analysis['high_conf_incorrect']}")

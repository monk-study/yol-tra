# Cell 1: Import required libraries
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2: Enhanced evaluation function
def evaluate_saved_model(model_path, test_loader, device='cuda'):
    """Comprehensive model evaluation"""
    # Load saved model
    saved_data = torch.load(model_path)
    
    # Recreate model architecture and load weights
    model = TPRRClassifier(
        input_dim=saved_data['input_dim'],
        hidden_dims=saved_data['model_config']['hidden_dims'],
        num_classes=saved_data['num_classes'],
        dropout_rate=saved_data['model_config']['dropout_rate']
    ).to(device)
    
    model.load_state_dict(saved_data['model_state_dict'])
    model.eval()
    
    # Collect predictions
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate basic metrics
    results = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'classification_report': classification_report(
            all_labels,
            all_preds,
            target_names=saved_data['label_encoder_classes'],
            output_dict=True
        ),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'probabilities': all_probs,
        'predictions': all_preds,
        'true_labels': all_labels
    }
    
    # Add per-class metrics
    class_names = saved_data['label_encoder_classes']
    per_class_metrics = {}
    
    for idx, class_name in enumerate(class_names):
        y_true_binary = (all_labels == idx)
        y_pred_binary = (all_preds == idx)
        
        per_class_metrics[class_name] = {
            'precision': precision_score(y_true_binary, y_pred_binary),
            'recall': recall_score(y_true_binary, y_pred_binary),
            'f1': f1_score(y_true_binary, y_pred_binary),
            'auc': roc_auc_score(y_true_binary, all_probs[:, idx])
        }
    
    results['per_class_metrics'] = per_class_metrics
    
    return results

# Cell 3: Enhanced prediction analysis
def analyze_predictions(results, threshold=0.9):
    """Analyze predictions with confidence levels"""
    high_confidence_correct = np.logical_and(
        np.max(results['probabilities'], axis=1) > threshold,
        results['predictions'] == results['true_labels']
    )
    
    low_confidence_correct = np.logical_and(
        np.max(results['probabilities'], axis=1) < 0.5,
        results['predictions'] == results['true_labels']
    )
    
    high_confidence_incorrect = np.logical_and(
        np.max(results['probabilities'], axis=1) > threshold,
        results['predictions'] != results['true_labels']
    )
    
    analysis = {
        'high_conf_correct': high_confidence_correct.sum(),
        'low_conf_correct': low_confidence_correct.sum(),
        'high_conf_incorrect': high_confidence_incorrect.sum()
    }
    
    # Print detailed analysis
    print("\nPrediction Analysis:")
    print(f"High confidence correct predictions: {analysis['high_conf_correct']}")
    print(f"Low confidence correct predictions: {analysis['low_conf_correct']}")
    print(f"High confidence incorrect predictions: {analysis['high_conf_incorrect']}")
    
    return analysis

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

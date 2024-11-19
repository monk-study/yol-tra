def evaluate_saved_model(model_path, test_loader, device='cpu'):
    """Comprehensive model evaluation"""
    # Load saved model
    saved_data = torch.load(model_path, map_location=device)
    
    # Recreate model architecture and load weights
    model = TPRRClassifier(
        input_dim=len(saved_data['feature_columns']),  # use length of feature columns
        hidden_dims=[256, 128],  # use same architecture as training
        num_classes=len(saved_data['label_encoder_classes']),
        dropout_rate=0.2  # use same as training
    ).to(device)
    
    model.load_state_dict(saved_data['model_state_dict'])
    model.eval()
    
    # Rest of the function remains same...
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
    
    return results, saved_data  # return saved_data as well for later use

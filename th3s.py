def get_branch_metrics(predictions, probabilities, labels):
    metrics = {}
    for branch in predictions.keys():
        y_true = np.array(labels[branch])
        y_pred = np.array(predictions[branch])
        y_prob = np.array(probabilities[branch])
        
        metrics[branch] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_prob)
        }
    
    return pd.DataFrame(metrics).T

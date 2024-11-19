

```
Branch Metrics:
          accuracy  precision  recall   f1      auc
NBA12    0.906951      0.0     0.0   0.0   0.500000
NBA3     0.696657      0.0     0.0   0.0   0.500000
NBA4     0.664904      0.0     0.0   0.0   0.500000
NBA5     0.921841      0.0     0.0   0.0   0.662381
NBA5CD   0.926648      0.0     0.0   0.0   0.500000
NBA7     0.930158      0.0     0.0   0.0   0.500000
NBA8     0.952841      0.0     0.0   0.0   0.500000

Val Loss: 0.3801, Val Acc: 0.8571
```



1. The high accuracies per branch (0.90, 0.69, etc.) but zero precision/recall suggest that our model is just predicting 0 for each branch
2. For each branch individually:
   - If NBA12 is 0 in 90.6% of cases, and we predict all 0s, we get 0.906951 accuracy
   - Similarly for NBA3: if it's 0 in 69.6% of cases, predicting all 0s gives 0.696657 accuracy
   - And so on...

3. Val Acc: 0.8571 was because val_acc = eval_results['branch_metrics']['accuracy'].mean()

By taking means of all branch accuracy
(0.906951 + 0.696657 + 0.664904 + 0.921841 + 0.926648 + 0.930158 + 0.952841) / 7 ≈ 0.8571

This is incorrect because:
1. Each sample should have exactly one NBA=1 (from data preprocessing)
2. If model predicts all zeros, it should be wrong for every sample
3. True accuracy should be 0% if predicting all zeros


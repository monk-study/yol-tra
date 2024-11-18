def validate_batch(model, features, labels, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        labels_onehot = F.one_hot(labels, num_classes=len(model.module.nba_branches)).float()
        loss = criterion(outputs, labels_onehot)
        predictions = (outputs > 0).float()
        accuracy = (predictions == labels_onehot).float().mean().item()
    return loss.item(), accuracy

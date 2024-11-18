class TPRRDataset(Dataset):
    def __init__(self, features, labels):
        # Remove any device assignment here - let DataLoader handle it
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

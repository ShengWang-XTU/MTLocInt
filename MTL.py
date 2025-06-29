
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        # Shared feature extractor
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        # Primary task: lncRNA multi-label subcellular localization prediction
        self.task1 = nn.Sequential(
            nn.Linear(64 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # The number of multi-label is 4
            nn.Sigmoid()
        )
        # Auxiliary task: lncRNA-protein interaction recognition
        self.task2 = nn.Sequential(
            nn.Linear(64 * 64 * 64 * 2, 128),  # Splicing lncRNA and protein features
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),  # Binary classification
            nn.Sigmoid()
        )

    def forward(self, x1, x2=None):
        # Shared feature extraction
        features1 = self.shared_cnn(x1)
        if x2 is not None:
            features2 = self.shared_cnn(x2)
            combined_features = torch.cat([features1, features2], dim=1)
            B = self.task2(combined_features)
            return B
        C = self.task1(features1)
        return C, None


# Custom Dataset Class
class MultiTask2Dataset(Dataset):
    def __init__(self, lncRNA_data, protein_data, labels):
        self.lncRNA_data = lncRNA_data
        self.protein_data = protein_data
        self.labels = labels

    def __len__(self):
        return len(self.lncRNA_data)

    def __getitem__(self, idx):
        lncRNA_sample = self.lncRNA_data[idx]
        protein_sample = self.protein_data[idx]
        label = self.labels[idx]
        return lncRNA_sample, protein_sample, label

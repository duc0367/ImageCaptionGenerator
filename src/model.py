import torch
import torch.nn as nn
from utils import build_batches
import os


# Remove fc, dropout
def build_image_encoder():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.fc = nn.Identity()
    model.dropout = nn.Identity()
    model.eval()
    return model


def encode_image(model, images: dict, batch_size: int) -> dict:
    keys = list(images.keys())
    values = list(images.values())
    batch_values = build_batches(values, batch_size)
    batch_results = []
    for batch in batch_values:
        batch_tensor = torch.tensor(batch, dtype=torch.float)
        batch_tensor = batch_tensor.permute(0, 3, 1, 2)
        batch_predictions = model(batch_tensor)
        batch_results.extend(batch_predictions.detach().numpy())

    result_dict = {}
    for key, value in zip(keys, batch_results):
        result_dict[key] = value
    return result_dict


class CaptionGeneratorModel(nn.Module):
    def __init__(self, vocab_size: int, image_feature: int, embeddings):
        super(CaptionGeneratorModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 200)
        self.embedding.weight = nn.Parameter(embeddings)
        self.embedding.requires_grad_(False)  # Disable train this layer
        self.dropout1 = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=200, hidden_size=256, num_layers=1, batch_first=True)

        self.dropout2 = nn.Dropout(0.3)
        self.linear1 = nn.Linear(image_feature, 256)

        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, vocab_size)

    def forward(self, x1, x2):
        # x1: (B, L)
        x1 = self.embedding(x1)  # (B, L, 200)
        x1 = self.dropout1(x1)  # (B, L, 200)
        x1, _ = self.lstm(x1)  # (B, L, 256)
        x1 = x1[:, -1, :]  # (B, 256) get the last time step

        # x2: (B, 2048)
        x2 = self.dropout2(x2)  # (B, 2048)
        x2 = self.linear1(x2)  # (B, 256)

        x = x1 + x2  # (B, 256)
        x = self.linear2(x)  # (B, 256)
        output = self.linear3(x)  # (B, Vocab_size)
        return output

    def save(self, filename='model.pth'):
        model_folder_path = '../model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        filename = os.path.join(model_folder_path, filename)
        torch.save(self.state_dict(), filename)

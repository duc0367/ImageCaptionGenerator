import torch
import torch.nn as nn
from utils import build_batches


# Remove fc, dropout and avgppol layer
def build_image_encoder():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.fc = nn.Identity()
    model.dropout = nn.Identity()
    model.avgpool = nn.Identity()
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
        print(batch_tensor.shape)
        batch_predictions = model(batch_tensor)
        print(batch_predictions.shape)
        batch_results.extend(batch_predictions.detach().numpy())

    result_dict = {}
    for key, value in zip(keys, batch_results):
        result_dict[key] = value
    return result_dict


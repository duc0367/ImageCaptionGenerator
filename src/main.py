import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import load_description, build_vocab, get_images, build_word_mapping, build_dataset,\
    get_max_length_description, build_embedding_matrix, generate_caption
from model import build_image_encoder, encode_image, CaptionGeneratorModel
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 10

LEARNING_RATE = 0.01

EPOCHS = 50

TEST_IMG = '667626_18933d713e'

descriptions = load_description()

vocab = build_vocab(descriptions)

image_ids = descriptions.keys()

images = get_images(image_ids)

model = build_image_encoder()

# dict of image key and its encoded values
encoded_images = encode_image(model, images, BATCH_SIZE)

id_to_word, word_to_id = build_word_mapping(vocab)

max_len = get_max_length_description(descriptions)

X1, X2, y = build_dataset(encoded_images, descriptions, word_to_id, max_len, BATCH_SIZE)

embeddings = build_embedding_matrix(word_to_id)

image_caption_generator = CaptionGeneratorModel(len(vocab), 2048, embeddings)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(image_caption_generator.parameters(), lr=LEARNING_RATE)

loss_vals = []

for epoch in range(EPOCHS):
    image_caption_generator.train()
    epoch_loss = []
    for batch_idx in tqdm(range(len(y))):
        x1_b = torch.tensor(X1[batch_idx], dtype=torch.float32).to(device)
        x2_b = torch.tensor(X2[batch_idx], dtype=torch.float32).to(device)
        y_b = torch.tensor(y[batch_idx], dtype=torch.float32).to(device)

        y_hat = image_caption_generator(x1_b, x2_b)
        loss = criterion(y_b, y_hat)
        epoch_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Inference
        if batch_idx + 1 % 100 == 0:
            print(f'Interation {batch_idx + 1}: {loss.item()}')
            print(f'Predict image: {TEST_IMG}')
            encoded_image = encoded_images[TEST_IMG]
            prediction = generate_caption(model, encoded_image, max_len, word_to_id, id_to_word)
            print(f'Predicted caption: {prediction}')
    # Record loss per each epoch
    loss_vals.append(sum(epoch_loss) / len(epoch_loss))

# Plot the loss
plt.plot(np.linspace(1, EPOCHS, EPOCHS).astype(int), loss_vals)

image_caption_generator.save()

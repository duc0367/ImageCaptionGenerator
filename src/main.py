import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_description, build_vocab, get_images, build_word_mapping, build_dataset,\
    get_max_length_description, build_embedding_matrix
from model import build_image_encoder, encode_image, CaptionGeneratorModel
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 10

LEARNING_RATE = 0.01

EPOCHS = 50

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

for epoch in range(EPOCHS):
    image_caption_generator.train()

    for batch_idx in tqdm(range(len(y))):
        x1_b = torch.tensor(X1[batch_idx], dtype=torch.float32).to(device)
        x2_b = torch.tensor(X2[batch_idx], dtype=torch.float32).to(device)
        y_b = torch.tensor(y[batch_idx], dtype=torch.float32).to(device)

        y_hat = image_caption_generator(x1_b, x2_b)
        loss = criterion(y_b, y_hat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

image_caption_generator.save()

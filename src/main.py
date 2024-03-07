from utils import load_description, build_vocab, get_images
from model import build_image_encoder, encode_image

BATCH_SIZE = 10

descriptions = load_description()

vocab = build_vocab(descriptions)

image_ids = descriptions.keys()

images = get_images(image_ids)

model = build_image_encoder()

# dict of image key and its encoded values
encoded_images = encode_image(model, images, BATCH_SIZE)
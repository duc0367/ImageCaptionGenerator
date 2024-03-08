import os
import string
import cv2
import numpy as np

BASE_PATH = '../data/'
VOCAB_THRESHOLD = 2
IMG_HEIGHT = 300
IMG_WIDTH = 300


# Remove punctual and not alpha text
def preprocessing_text(text: str):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.split()

    new_text = []
    for word in text:
        if word.isalpha():
            new_text.append(word.lower())

    return ' '.join(new_text)


def add_start_stop_word(text: str):
    return 'startseq ' + text + ' endseq'


def load_description():
    f = open(os.path.join(BASE_PATH, "FlicrkData", "Flickr8k.token.txt"), "r")
    lines = f.readlines()
    lines = lines[:12]
    descriptions = {}

    for line in lines:
        line = line.split('\t')
        id_with_index = line[0].split('#')
        image_id = id_with_index[0]
        image_id = image_id[0: len(image_id)-4]
        description = preprocessing_text(line[1])
        description = add_start_stop_word(description)
        if image_id not in descriptions:
            descriptions[image_id] = [description]
        else:
            descriptions[image_id].append(description)

    return descriptions


def build_vocab(descriptions: dict):
    vocab = set()
    count = {}
    for values in descriptions.values():
        for des in values:
            words = des.split()
            for word in words:
                if word not in count:
                    count[word] = 1
                else:
                    count[word] += 1

                if count[word] >= VOCAB_THRESHOLD:
                    vocab.add(word)
    vocab.add('OOV')
    return vocab


def get_images(image_ids):
    images = {}
    for img_id in image_ids:
        image_data = cv2.imread(os.path.join(BASE_PATH, 'Images', img_id + '.jpg'))  # BGR order
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)  # RGB
        images[img_id] = preprocessing_img(image_data)
    return images


def preprocessing_img(image):
    return cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))


def build_batches(data, batch_size: int):
    num_samples = len(data)
    batches = []
    for i in range(0, len(data), batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch = data[i: end_idx]
        batches.append(batch)
    return np.array(batches)


def build_word_mapping(vocab: set):
    id_to_word = {}
    word_to_id = {}

    for idx, word in enumerate(vocab):
        id_to_word[idx] = word
        word_to_id[word] = idx

    return id_to_word, word_to_id


def build_dataset(encoded_images, descriptions: dict, word_to_idx: dict, max_len: int, batch_size):
    x1 = []
    x2 = []
    y = []
    image_ids = descriptions.keys()
    for img_id in image_ids:
        descriptions_per_img_id = descriptions[img_id]
        for description in descriptions_per_img_id:
            words = description.split()
            words = [word_to_idx[word] if word in word_to_idx else word_to_idx['OOV'] for word in words]
            for idx in range(1, len(words)):
                x2.append(encoded_images[img_id])
                padding_length = max(0, max_len - len(words[:idx]))
                x1.append(np.pad(words[:idx], (0, padding_length), 'constant', constant_values=(0,)))
                y.append(words[idx])
    x1 = batch_size(x1, batch_size)
    x2 = batch_size(x2, batch_size)
    y = batch_size(y, batch_size)
    return x1, x2, y


def get_max_length_description(descriptions: dict):
    descriptions_arr = list(descriptions.values())
    descriptions_flatten = [
            x
            for xs in descriptions_arr
            for x in xs
    ]
    descriptions_length = [len(description.split()) for description in descriptions_flatten]
    return np.max(descriptions_length)


def get_embedding_index_from_glove() -> dict:
    glove_path = os.path.join(BASE_PATH, 'glove.6B.200d.txt')
    glove = open(glove_path, 'r', encoding='utf-8').read()
    embedding_index = {}
    for line in glove.split("\n"):
        values = line.split(" ")
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = embedding
    return embedding_index


def build_embedding_matrix(word_to_id: dict):
    embeddings_index = get_embedding_index_from_glove()
    vocab_size = len(word_to_id)
    embedding_matrix = np.zeros((vocab_size, 200))
    for word, idx in word_to_id:
        embedding_word = embeddings_index.get(word)
        if embedding_word is not None:
            embedding_matrix[idx] = embedding_word
    return embedding_matrix

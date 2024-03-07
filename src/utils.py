import os
import string
import cv2

BASE_PATH = '../data/'
VOCAB_THRESHOLD = 10
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
    lines = lines[:6]
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
    return batches

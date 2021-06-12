import random
from pickle import load
from keras_preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import numpy as np


img_train_path = "./Dataset/Flickr8k_text/Flickr_8k.trainImages.txt"
img_test_path = "./Dataset/Flickr8k_text/Flickr_8k.devImages.txt"
img_features_path = "./training_files/img_features.pkl"
captions_filename = "./training_files/captions.txt"


def load_img_ids(filename):
    """
    Load image ids from a file
    """
    file = open(filename, "r")
    text = file.read()
    file.close()

    img_ids = list()
    for line in text.split("\n"):

        if len(line) < 1:
            continue

        img_id = line.split('.')[0]
        img_ids.append(img_id)

    return img_ids


def load_img_features(img_features, train_ids, test_ids):
    """
    Load train and test features from a file
    :param img_features:
    :param train_ids:
    :param test_ids:
    :return:
    """
    features = load(open(img_features, "rb"))

    train_features = {train_id: features[train_id] for train_id in train_ids}
    test_features = {test_id: features[test_id] for test_id in test_ids}

    return train_features, test_features


def load_clean_captions(filename, dataset):
    """
    load captions from file and create entry for each imgId from dataset
    """
    file = open(filename, 'r')
    text = file.read()
    file.close()

    captions = dict()

    for line in text.split('\n'):

        tokens = line.split()
        img_id, img_caption = tokens[0], tokens[1:]

        if img_id in dataset:
            if img_id not in captions:
                captions[img_id] = list()

            # add startseq at the begining and endseq at the end of each caption
            caption = 'startseq ' + ' '.join(img_caption) + ' endseq'
            captions[img_id].append(caption)

    return captions


def load_train_test(img_features_path, captions_path, train_ids_path, test_ids_path):
    """
    Load train image features and captions, load test image features and captions
    :param img_features_path:
    :param captions_path:
    :param train_ids_path:
    :param test_ids_path:
    :return:
    """
    img_train_ids = load_img_ids(img_train_path)
    img_test_ids = load_img_ids(img_test_path)

    train_features, test_features = load_img_features(img_features_path, img_train_ids, img_test_ids)
    print("Train images: ", len(train_features))
    print("Test images: ", len(test_features))

    train_captions = load_clean_captions(captions_filename, img_train_ids)
    test_captions = load_clean_captions(captions_filename, img_test_ids)

    return train_features, train_captions, test_features, test_captions

train_features, train_captions, test_features, test_captions = load_train_test(img_features_path, captions_filename, img_train_path, img_test_path)

# ///////////////////////////////////////////////////////////////////////////////////////// #


def to_lines(captions):
    """
    Extract values from captions dictionary
    """
    all_captions = list()
    for key in captions.keys():
        [all_captions.append(d) for d in captions[key]]
    return all_captions


def create_tokenizer(captions):
    lines = to_lines(captions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def max_length(captions):
    lines = to_lines(captions)
    return max(len(line.split()) for line in lines)


tokenizer = create_tokenizer(train_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max_length(train_captions)
print(max_length)


def create_sequences(image, caption_list, tokenizer, max_length, vocab_size):
    """
    Generate sequences from a caption, containing just the first word, first two words etc.
    For word i in sequence, separate the caption into input=caption[:i] and next_word=caption[i];
    encode each word as a categorical value.
    :param image:
    :param caption_list:
    :param tokenizer:
    :param max_length:
    :param vocab_size:
    :return:
    """
    in_img_list, in_word_list, out_word_list = list(), list(), list()
    for capt in caption_list:
        # tokenize each caption
        seq = tokenizer.texts_to_sequences([capt])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode word to a categorical value
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

            in_img_list.append(image)
            in_word_list.append(in_seq)
            out_word_list.append(out_seq)
    return in_img_list, in_word_list, out_word_list


def data_generator(images, captions, tokenizer, max_length, batch_size, random_seed, vocab_size):
    """
    Extract images, input word sequences and output word in batches. To be used while fitting the model.
    :param images:
    :param captions:
    :param tokenizer:
    :param max_length:
    :param batch_size:
    :param random_seed:
    :param vocab_size:
    :return:
    """
    random.seed(random_seed)

    img_ids = list(captions.keys())

    count = 0
    while True:
        if count >= len(img_ids):
            count = 0

        in_img_batch, in_seq_batch, out_word_batch = list(), list(), list()

        # get current batch indexes
        for i in range(count, min(len(img_ids), count+batch_size)):
            # current image_id
            img_id = img_ids[i]
            # current image
            img = images[img_id][0]
            # current image caption list
            captions_list = captions[img_id]
            # shuffle the captions
            random.shuffle(captions_list)
            # get word sequences and output word
            in_img, in_seq, out_word = create_sequences(img, captions_list, tokenizer, max_length, vocab_size)

            # append to batch list
            for j in range(len(in_img)):
                in_img_batch.append(in_img[j])
                in_seq_batch.append(in_seq[j])
                out_word_batch.append(out_word[j])

        count = count + batch_size
        yield [[np.array(in_img_batch), np.array(in_seq_batch), np.array(out_word_batch)]]

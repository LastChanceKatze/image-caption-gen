from pickle import load


def load_img_ids(filename):
    # read from the captions file
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
    img_train_ids = load_img_ids(train_ids_path)
    img_test_ids = load_img_ids(test_ids_path)

    train_features, test_features = load_img_features(img_features_path, img_train_ids, img_test_ids)

    train_captions = load_clean_captions(captions_path, img_train_ids)
    test_captions = load_clean_captions(captions_path, img_test_ids)

    print("Train images: ", len(train_features))
    print("Train captions: ", len(train_captions))
    print("Test images: ", len(test_features))
    print("Test captions: ", len(test_captions))

    return train_features, train_captions, test_features, test_captions

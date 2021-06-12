from os import listdir
import string
from pickle import dump
import tensorflow.keras.applications.vgg16 as vgg16
import tensorflow.keras.applications.inception_v3 as inception_v3
from tensorflow.keras.models import Model
from keras.preprocessing.image import load_img, img_to_array


def load_captions(filename):
    """
    Load captions from file and create a per image caption dictionary
    :param filename:
    :return:
    """
    # read from the captions file
    file = open(filename, "r")
    text = file.read()
    file.close()

    mapping = dict()

    # process each line
    # line is in form: image_name.jpg#no caption
    for line in text.split("\n"):
        token = line.split("\t")

        if len(line) < 2:
            continue

        # first token: image id
        # rest: image caption
        img_id, img_capt = token[0], token[1:]
        # extract image id: before the .jpg part
        img_id = img_id.split('.')[0]
        # convert caption list back to string
        img_capt = ' '.join(img_capt)

        # add all the captions od the same image to image_id key
        if img_id not in mapping:
            mapping[img_id] = list()
        mapping[img_id].append(img_capt)

        print(img_id)

    return mapping


def clean_captions(captions):
    """
    Remove punctuation, hanging s and a, and tokens with numbers
    from the captions
    :param captions:
    :return:
    """
    # Prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for _, caption_list in captions.items():
        for i in range(len(caption_list)):
            caption = caption_list[i]
            # Tokenize i.e. split on white spaces
            caption = caption.split()
            # Convert to lowercase
            caption = [word.lower() for word in caption]
            # Remove punctuation from each token
            caption = [w.translate(table) for w in caption]
            # Remove hanging 's' and 'a'
            caption = [word for word in caption if len(word)>1]
            # Remove tokens with numbers in them
            caption = [word for word in caption if word.isalpha()]
            # Store as string
            caption_list[i] = ' '.join(caption)


def save_captions(captions_dict, to_file):
    """
    Save the captions_dict to a file,
    file: image_id caption_list per line
    :param captions_dict:
    :param to_file:
    :return:
    """
    # convert captions dictionary to string of lines
    lines = list()
    for key, caption_list in captions_dict.items():
        for caption in caption_list:
            lines.append(key + ' ' + caption)
    data = '\n'.join(lines)

    # save captions string to a file
    file = open(to_file, 'w')
    file.write(data)
    file.close()


def preprocess_captions(capt_filename="./Dataset/Flickr8k_text/Flickr8k.lemma.token.txt",
                        clean_capt_to_file="./training_files/captions.txt"):
    captions_dict = load_captions(capt_filename)
    clean_captions(captions_dict)
    save_captions(captions_dict, clean_capt_to_file)


def create_cnn_model_dict():
  cnn_model_dict = dict()

  cnn_model_dict['vgg16'] = {
      'model': vgg16.VGG16(),
      'target_size': (224, 224),
      'preprocess_input': vgg16.preprocess_input
  }

  cnn_model_dict['inception_v3'] = {
      'model': inception_v3.InceptionV3(),
      'target_size': (299, 299),
      'preprocess_input': inception_v3.preprocess_input
  }
  return cnn_model_dict


def extract_features(images_dir, model_type, cnn_model_dict):
    model = cnn_model_dict[model_type]['model']
    target_size = cnn_model_dict[model_type]['target_size']
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    model.summary()

    features_dict = dict()

    img_count = 0

    for name in listdir(images_dir):
        filename = f"{images_dir}/{name}"
        image = load_img(filename, target_size=target_size)
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = cnn_model_dict[model_type]['preprocess_input'](image)
        features = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features_dict[image_id] = features

        img_count += 1

        if img_count % 200 == 0:
            print("No. images", img_count)
            print()

        print(".", end="")

    return features_dict


def save_img_features(img_features, to_file):
    dump(img_features, open(to_file, "wb"))


def preprocess_img_features(model_type, images_dir=f"./Dataset/Flickr8k_Dataset/Flicker8k_Dataset",
                            to_file=f"./training_files/img_features.pkl"):
    cnn_model_dict = create_cnn_model_dict()
    features = extract_features(images_dir, model_type, cnn_model_dict)
    print("No. features", len(features))
    save_img_features(features, to_file)


import string


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




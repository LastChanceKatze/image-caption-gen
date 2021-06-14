from tensorflow.keras.models import Model
from keras_preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import matplotlib.pyplot as plt
import params_dict

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


def calc_max_length(captions):
    lines = to_lines(captions)
    return max(len(line.split()) for line in lines)


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
        for i in range(count, min(len(img_ids), count + batch_size)):
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
        yield [np.array(in_img_batch), np.array(in_seq_batch)], np.array(out_word_batch)


# define the captioning model
def define_model(vocab_size, max_length):
    image_input = Input(shape=(4096,))
    image_model_1 = Dropout(0.5)(image_input)
    image_model = Dense(256, activation='relu')(image_model_1)

    caption_input = Input(shape=(max_length,))
    # mask_zero: We zero pad inputs to the same length, the zero mask ignores those inputs. E.g. it is an efficiency.
    caption_model_1 = Embedding(vocab_size, 256, mask_zero=True)(caption_input)
    caption_model_2 = Dropout(0.5)(caption_model_1)
    caption_model = LSTM(256)(caption_model_2)

    # Merging the models and creating a softmax classifier
    final_model_1 = concatenate([image_model, caption_model])
    final_model_2 = Dense(256, activation='relu')(final_model_1)
    final_model = Dense(vocab_size, activation='softmax')(final_model_2)

    model = Model(inputs=[image_input, caption_input], outputs=final_model)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model


def plot_history(history):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss through epochs')
    plt.legend(['Train', 'Test'], loc='best')


# TODO: should add necessary params in order to work
def train_model(params):
    train_steps = len(params['train_captions']) // params['batch_size']
    if len(params['train_captions']) % params['batch_size'] != 0:
        train_steps = train_steps + 1

    test_steps = len(params['test_captions']) // params['batch_size']
    if len(params['test_captions']) % params['batch_size'] != 0:
        test_steps = test_steps + 1

    checkpoint = ModelCheckpoint(params['filepath'], monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    train_generator = data_generator(params['train_features'], params['train_captions'], params['tokenizer'],
                                     params['max_length'], params['batch_size'], 10, params['vocab_size'])
    test_generator = data_generator(params['test_features'], params['test_captions'], params['tokenizer'],
                                    params['max_length'], params['batch_size'], 10, params['vocab_size'])

    model = params['model']
    history = model.fit(train_generator, epochs=params['epochs'], steps_per_epoch=train_steps,
                        validation_data=test_generator, validation_steps=test_steps,
                        callbacks=[checkpoint], verbose=1)

    if params['plot_hist']:
        plot_history(history.history)

    return model


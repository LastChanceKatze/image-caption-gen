import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu


def index_to_word(word_index, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == word_index:
            return word
    return None


def generate_caption(model, tokenizer, image, max_length):
    in_caption = "startseq"

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_caption])[0]

        seq = pad_sequences([seq], maxlen=max_length)

        pred = model.predict([image, seq], verbose=0)

        pred = np.argmax(pred)

        word = index_to_word(pred, tokenizer)

        if word is None:
            break

        in_caption += ' ' + word

        if word == 'endseq':
            break

    return in_caption


def evaluate_model(model, images, captions, tokenizer, max_lenght):
    test, predicted = list(), list()

    img_count = 0
    for key, caption_list in captions.items():
        pred = generate_caption(model, tokenizer, images[key], max_lenght)

        predicted.append(pred.split())
        test.append([capt.split() for capt in caption_list])

        img_count += 1

        if img_count % 200 == 0:
            print("No. images", img_count)
            print()

        print(".", end="")

    print("BLEU-1 ", corpus_bleu(test, predicted, weights=(1.0, 0, 0, 0)))
    print("BLEU-2 ", corpus_bleu(test, predicted, weights=(.5, .5, 0, 0)))
    print("BLEU-3 ", corpus_bleu(test, predicted, weights=(.3, .3, .3, 0)))
    print("BLEU-4 ", corpus_bleu(test, predicted))


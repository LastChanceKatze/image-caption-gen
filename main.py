import preprocess as ld
import load_data as load_data
import os
import create_sequences as cs

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.applications.vgg16 import VGG16

def main():
    # ld.preprocess_captions()
    # load.load_train_test()
    captions_filename = './training_files/captions.txt'
    img_features_path = './training_files/img_features.pkl'
    img_train_path = './Dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
    img_test_path = './Dataset/Flickr8k_text/Flickr_8k.devImages.txt'
    glove_path = './glove/glove.6B.200d.txt'

    train_features, train_captions, test_features, test_captions = load_data.load_train_test(img_features_path,
                                                                                             captions_filename,
                                                                                             img_train_path,
                                                                                             img_test_path)

    tokenizer = cs.create_tokenizer(train_captions)
    vocab_size = len(tokenizer.word_index) + 1
    #all_training_captions = cs.to_lines(test_captions)
    #print(all_training_captions)
    #print(load_data.get_vocab(all_training_captions))

    embedding = load_data.glove_embedding_indices(glove_path)
    embedding_matrix = load_data.get_vocab_embedding_weights(embedding, tokenizer.word_index, vocab_size)
    print(embedding_matrix)

    train_model_params = {'model': model,
                          'epochs': 20,
                          'batch_size': 64,
                          'plot_hist': True,
                          'train_captions': train_captions,
                          'test_captions': test_captions,
                          'filepath': drive_folder + "/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                          'tokenizer': tokenizer,
                          'max_length': max_length,
                          'vocab_size': vocab_size,
                          'train_features': train_features,
                          'test_features': test_features
                          }

if __name__ == '__main__':
    main()

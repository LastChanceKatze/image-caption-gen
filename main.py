import preprocess as ld
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.applications.vgg16 import VGG16


def main():
    ld.preprocess_captions()


if __name__ == '__main__':
    main()

# baseline model with dropout and data augmentation on the cifar10 dataset
from keras.models import load_model
import os

# define cnn model
def CIFAR10_final_model():
    model = load_model(os.path.join('%smodels/weights' % './', "final_model_100epochs.h5"))
    # model = load_model(os.path.join('%smodels/weights' % './', "final_model543.h5"))
    return model
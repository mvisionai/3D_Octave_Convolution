import tensorflow as tf
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import LearningRateScheduler, History
from models import *
from AD_Dataset import  Dataset_Import
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Model, load_model

from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils import to_categorical
import pickle, os, time
import numpy as np
from sklearn.utils import shuffle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,batch_size,source_data,shuffle_data=False):
        'Initialization'
        self.batch_size=batch_size
        self.source_data =source_data
        self.shuffle_data=shuffle_data
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.source_data) / self.batch_size))


    def __getitem__(self,index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #print("My index ",index)
        # Generate data
        batch_data = self.source_data[index * self.batch_size:(index + 1) * self.batch_size]
        #print(batch_data)
        X, y = self.__data_generation(batch_data)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle_data ==True:
            shuffle(self.source_data)

    def __data_generation(self,batch_data):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        datafeed = Dataset_Import()
        #validation_datas = shuffle(datafeed.all_main_validate())[0:3]
        return  np.array(datafeed.convert_batch_to_img_dataK(batch_data)), np.array(datafeed.convert_batch_label_dataK(batch_data))

def save_best_model(epoch, dir_path, num_ext, ext):
    tmp_file_name = os.listdir(dir_path)
    test = []
    num_element = -num_ext

    for x in range(0, len(tmp_file_name)):
	test.append(tmp_file_name[x][:num_element])
	float(test[x])

    highest = max(test)

    return str(highest) + ext


def lr_scheduler(epoch):
    x = 0.1
    if epoch >= 100: x /= 5.0
    if epoch >= 150: x /= 5.0
    return x

def train(alpha):
    # (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # train_gen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True,
    #                                 width_shift_range=4.0/32.0, height_shift_range=4.0/32.0)
    # test_gen = ImageDataGenerator(rescale=1.0/255)
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    # Data feed importation for trainging
    datafeeds = Dataset_Import()
    source_data_feed = shuffle(datafeeds.all_source_data(augment_data=True))
    validation_data_feed = shuffle(datafeeds.all_main_validate())

    gen_source = DataGenerator(batch_size=1, source_data=source_data_feed, shuffle_data=False)
    gen_validation = DataGenerator(batch_size=1, source_data=validation_data_feed, shuffle_data=False)

    checkpoint = ModelCheckpoint('output/{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='auto')
    if alpha <= 0:
        model = create_normal_wide_resnet()
    else:
        model = create_octconv_wide_resnet(alpha)
    model.compile(SGD(0.00001, momentum=0.9), "categorical_crossentropy", ["acc"])
    model.summary()


    batch_size = 1
    #scheduler = LearningRateScheduler(lr_scheduler)
    hist = History()

    start_time = time.time()
    model.fit_generator(generator=gen_source,
                        steps_per_epoch=len(source_data_feed),
                        validation_data=gen_validation,
                        validation_steps=len(validation_data_feed),
                        callbacks=[hist,checkpoint], max_queue_size=5, epochs=200)


    elapsed = time.time() - start_time
    print(elapsed)

    history = hist.history
    history["elapsed"] = elapsed

    best = save_best_model(10, "output", 5, ".hdf5")
    model.load_weights(best)
    model.save("best.h5s")

    with open("octconv_alpha_"+alpha+".pkl", "wb") as fp:
        pickle.dump(history, fp)

if __name__ == "__main__":
    train(0.5)


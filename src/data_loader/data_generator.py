import numpy as np
from src.utils.utils import unpickle
from src.data_loader.preprocessing import get_labels, paths_to_images
from random import shuffle
from glob import glob



class DataGenerator:
    """DataGenerator class responsible for dealing with cifar-100 dataset.

    Attributes:
        config: Config object to store data related to training, testing and validation.
        all_train_data: Contains the whole dataset(since the dataset fits in memory).
        x_all_train: Contains  the whole input training-data.
        x_all_train: Contains  the whole target_output labels for training-data.
        x_train: Contains training set inputs.
        y_train: Contains training set target output.
        x_val: Contains validation set inputs.
        y_val: Contains validation set target output.
        meta: Contains meta-data about Cifar-100 dataset(including label names).
    """

    def __init__(self, config, training=True, testing=False):
        self.config = config
        self.training = training
        self.testing = testing
        self.all_train_data = None
        self.x_all_train = None
        self.y_all_train = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

        self.num_batches_train = None
        self.num_batches_val = None
        self.num_batches_test = None

        self.indx_batch_train = 0
        self.indx_batch_val = 0
        self.indx_batch_test = 0

        if self.training:
            self.__load_train_data()
            self.__load_validation_data()
        if self.testing:
            self.__load_test_data()


    def __load_test_data(self):
        """Private function.
        Returns:
        """
        self.x_test = np.asanyarray(glob(self.config.test_data_path+"*"))
        # print("x test: ",self.x_test)
        self.y_test = np.asanyarray(get_labels(self.x_test))
        self.num_batches_test = int(np.ceil(self.x_test.shape[0] / self.config.batch_size))

    def __load_validation_data(self):
        """Private function.
        Returns:
        """
        self.x_val = np.asanyarray(glob(self.config.val_data_path+"*"))
        self.y_val = np.asanyarray(get_labels(self.x_val))
        self.num_batches_val = int(np.ceil(self.x_val.shape[0] / self.config.batch_size))



    def __load_train_data(self):
        """Private function.
        Returns:
        """
        self.x_all_train = np.asanyarray(glob(self.config.train_data_path+"*"))
        self.y_all_train = np.asanyarray(get_labels(self.x_all_train))

        self.__shuffle_all_data()
        # self.__split_train_val()
        self.x_train = self.x_all_train
        self.y_train = self.y_all_train
        self.num_batches_train = int(np.ceil(self.x_train.shape[0] / self.config.batch_size))
        # self.num_batches_val = int(np.ceil(self.x_val.shape[0] / self.config.batch_size))
        print("total training examples:", self.x_train.shape[000])

    def __shuffle_all_data(self):
        """Private function.
        Shuffles the whole training set to avoid patterns recognition by the model(I liked that course:D).
        shuffle function is used instead of sklearn shuffle function in order reduce usage of
        external dependencies.
        ***Note:
            Dataset is BGR format not RGB!..

        Returns:
        """
        indices_list = [i for i in range(self.x_all_train.shape[0])]
        shuffle(indices_list)
        # Next two lines may cause memory error if no sufficient ram.
        self.x_all_train = self.x_all_train[indices_list]
        self.y_all_train = self.y_all_train[indices_list]

    # def __split_train_val(self):
    #     """Private function.
    #     Splits the training set to train and validation sets using config.val_split_ratio from config file.
    #     ***Note:
    #         Dataset is BGR format not RGB!..
    #
    #     Returns:
    #     """
    #     if self.config.use_val:
    #         split_point = int(self.config.val_split_ratio * self.x_all_train.shape[0])
    #     else:
    #         split_point = 0
    #     self.x_train = self.x_all_train[split_point:self.x_all_train.shape[0]]
    #     self.y_train = self.y_all_train[split_point:self.y_all_train.shape[0]]
    #     self.x_val = self.x_all_train[0:split_point]
    #     self.y_val = self.y_all_train[0:split_point]

    def __shuffle_train_data(self):
        """Private function.
        Shuffles the training data.
        TODO(MohamedAli1995): Remove this function and build a base class to inherit from, this is
        a short-term solution, better build a hierarchical OOP structure.
        ***Note:
            Dataset is BGR format not RGB!..

        Returns:
        """
        indices_list = [i for i in range(self.x_train.shape[0])]
        shuffle(indices_list)

        # Next two lines may cause memory error if no sufficient ram.
        self.x_train = self.x_train[indices_list]
        self.y_train = self.y_train[indices_list]

    def prepare_new_epoch_data(self):
        """Prepares the dataset for a new epoch by setting the indx of the batches to 0 and shuffling
        the training data.

        Returns:
        """
        self.indx_batch_train = 0
        self.indx_batch_val = 0
        self.indx_batch_test = 0
        self.__shuffle_train_data()

    def next_batch(self, batch_type="train"):
        """Moves the indx_batch_... pointer to the next segment of the data.

        Args:
            batch_type: the type of the batch to be returned(train, test, validation).

        Returns:
            The next batch of the data with type of batch_type.
        """
        if batch_type == "train":
            x = self.x_train[self.indx_batch_train:self.indx_batch_train + self.config.batch_size]
            y = self.y_train[self.indx_batch_train:self.indx_batch_train + self.config.batch_size]
            self.indx_batch_train = (self.indx_batch_train + self.config.batch_size) % self.x_train.shape[0]

        elif batch_type == "val":
            x = self.x_val[self.indx_batch_val:self.indx_batch_val + self.config.batch_size]
            y = self.y_val[self.indx_batch_val:self.indx_batch_val + self.config.batch_size]
            self.indx_batch_val = (self.indx_batch_val + self.config.batch_size) % self.x_val.shape[0]
        elif batch_type == "test":
            x = self.x_test[self.indx_batch_test:self.indx_batch_test + self.config.batch_size]
            y = self.y_test[self.indx_batch_test:self.indx_batch_test + self.config.batch_size]
            self.indx_batch_test = (self.indx_batch_test + self.config.batch_size) % self.x_test.shape[0]

        x = paths_to_images(x, self.config.state_size)
        return x, y
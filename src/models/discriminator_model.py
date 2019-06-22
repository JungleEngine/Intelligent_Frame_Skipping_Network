from src.base.base_model import BaseModel
import tensorflow as tf


class DiscriminatorModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.is_training = None
        self.x = None
        self.y = None
        self.train_step = None
        self.saver = None
        self.hold_prob = None

        self.build_model()
        self.init_saver()

    def __init_weights(self, shape):
        initializer = tf.contrib.layers.xavier_initializer()
        init_xavier = initializer(shape)
        return tf.Variable(init_xavier)

    def __init_bias(self, shape):
        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals)


    def __max_pool_2d(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def __convolution_layer(self, input_x, shape, strides=None):
        if strides is None:
            strides = [1, 1, 1, 1]
        W = self.__init_weights(shape)
        b = self.__init_bias([shape[3]])
        return tf.nn.conv2d(input_x, W, strides=strides, padding='SAME') + b

    def __normal_full_layer(self, input_layer, size):
        input_size = int(input_layer.get_shape()[1])  # becase 0 is the number of training examples.
        W = self.__init_weights([input_size, size])
        b = self.__init_bias([size])
        return tf.matmul(input_layer, W) + b

    def __batch_norm(self, input_layer):
        return tf.contrib.layers.batch_norm(input_layer, activation_fn=tf.nn.leaky_relu, is_training=self.is_training)

    def __conv_bn_layer(self, input_layer, use_bn, shape, strides):
        _conv = self.__convolution_layer(input_layer, shape=shape, strides=strides)
        if use_bn:
            return self.__batch_norm(_conv)
        else:
            return tf.nn.leaky_relu(_conv)

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.hold_prob = tf.placeholder(tf.float32)

        convo_1 = self.__convolution_layer(self.x, shape=[5, 5, 6, 32])
        convo_1_pooling = self.__max_pool_2d(convo_1)
        batch_norm_1 = self.__batch_norm(convo_1_pooling)
        dropout_1 = tf.nn.dropout(batch_norm_1, self.hold_prob)

        convo_2 = self.__convolution_layer(dropout_1, shape=[5, 5, 32, 64])
        convo_2_pooling = self.__max_pool_2d(convo_2)
        batch_norm_2 = self.__batch_norm(convo_2_pooling)
        dropout_2 = tf.nn.dropout(batch_norm_2, self.hold_prob)

        convo_3 = self.__convolution_layer(dropout_2, shape=[3, 3, 64, 128])
        convo_3_pooling = self.__max_pool_2d(convo_3)
        batch_norm_3 = self.__batch_norm(convo_3_pooling)
        dropout_3 = tf.nn.dropout(batch_norm_3, self.hold_prob)

        convo_4 = self.__convolution_layer(dropout_3, shape=[3, 3, 128, 256])
        convo_4_pooling = self.__max_pool_2d(convo_4)
        batch_norm_4 = self.__batch_norm(convo_4_pooling)
        dropout_4 = tf.nn.dropout(batch_norm_4, self.hold_prob)

        convo_5 = self.__convolution_layer(dropout_4, shape=[1, 1, 256, 512])
        convo_5_pooling = self.__max_pool_2d(convo_5)
        batch_norm_5 = self.__batch_norm(convo_5_pooling)
        dropout_5 = tf.nn.dropout(batch_norm_5, self.hold_prob)

        flattened = tf.reshape(dropout_5,
                               [-1, 8 * 8 * 512])

        full_layer_1 = self.__normal_full_layer(flattened, 1024)
        batch_norm_6 = self.__batch_norm(full_layer_1)
        full_dropout_1 = tf.nn.dropout(batch_norm_6, self.hold_prob)

        full_layer_2 = self.__normal_full_layer(full_dropout_1, 2048)
        batch_norm_7 = self.__batch_norm(full_layer_2)
        full_dropout_2 = tf.nn.dropout(batch_norm_7, self.hold_prob)


        self.y_pred = self.__normal_full_layer(full_dropout_2, 1)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.y_pred))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                              global_step=self.global_step_tensor)
            correct_predictions = tf.equal(tf.round(tf.nn.sigmoid(self.y_pred)), tf.round(self.y))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            self.predictions = tf.round(tf.nn.sigmoid(self.y_pred))
            self.predictions_scores = tf.nn.sigmoid(self.y_pred)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)

from src.base.base_model import BaseModel
import tensorflow as tf


class DiscriminatorModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.is_training = None
        self.x1 = None
        self.x2 = None
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
        init_bias_vals = tf.constant(0.01, shape=shape)
        return tf.Variable(init_bias_vals)


    def __max_pool_2d(self, x, ksize=None, strides=None):
        if strides is None:
            strides = [1, 2, 2, 1]
        if ksize is None:
            ksize = [1, 2, 2, 1]
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='VALID')
    
    def __average_pool_2d(self, x):
        return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    def __convolution_layer(self, input_x, shape, padding='VALID', strides=None):
        if strides is None:
            strides = [1, 1, 1, 1]
        W = self.__init_weights(shape)
        b = self.__init_bias([shape[3]])
        return tf.nn.conv2d(input_x, W, strides=strides, padding=padding) + b

    def __normal_full_layer(self, input_layer, size):
        input_size = int(input_layer.get_shape()[1])  # becase 0 is the number of training examples.
        W = self.__init_weights([input_size, size])
        b = self.__init_bias([size])
        return tf.matmul(input_layer, W) + b

    def __batch_norm(self, input_layer):
        return tf.contrib.layers.batch_norm(input_layer, activation_fn=tf.nn.relu,
                                            is_training=tf.cast(self.is_training, tf.bool))

    def __conv_bn_layer(self, input_layer, shape, padding='VALID', strides=None, use_bn=True):
        if strides is None:
            strides = [1, 1, 1, 1]
        _conv = self.__convolution_layer(input_layer, shape=shape, padding=padding, strides=strides)
        if use_bn:
            return self.__batch_norm(_conv)
        else:
            return tf.nn.relu(_conv)

    def build_model(self):
        self.is_training = tf.placeholder_with_default(0, shape=(), name="is_training")

        self.x1 = tf.placeholder(tf.float32, shape=[None] + self.config.state_size, name="input_1")
        self.x2 = tf.placeholder(tf.float32, shape=[None] + self.config.state_size, name="input_2")

        concat = tf.concat([self.x1, self.x2], -1)

        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.hold_prob_conv = tf.placeholder_with_default(1.0,shape=(), name="hold_prob_conv")
        self.hold_prob_fc = tf.placeholder_with_default(1.0,shape=(), name="hold_prob_fc")

        convo_1 = self.__conv_bn_layer(concat, shape=[7, 7, 6, 96], strides=[1, 4, 4, 1], padding='VALID')
        dropout_1 = tf.nn.dropout(convo_1, self.hold_prob_conv)
        convo_1_pooling = self.__max_pool_2d(dropout_1, ksize=[1, 3, 3, 1])
        
        convo_2 = self.__conv_bn_layer(convo_1_pooling, shape=[5, 5, 96, 256], padding='SAME')
        dropout_2 = tf.nn.dropout(convo_2, self.hold_prob_conv)
        convo_2_pooling = self.__max_pool_2d(dropout_2, ksize=[1, 3, 3, 1])
        
        convo_3 = self.__conv_bn_layer(convo_2_pooling, shape=[3, 3, 256, 384], padding='SAME')
        dropout_3 = tf.nn.dropout(convo_3, self.hold_prob_conv)
        convo_3_pooling = self.__max_pool_2d(dropout_3, ksize=[1, 3, 3, 1])
        
        flattened = tf.reshape(convo_3_pooling,
                               [-1, 6 * 6 * 384])
                
        self.y_pred = self.__normal_full_layer(flattened, 1)
        
        self.predictions = tf.round(tf.nn.sigmoid(self.y_pred), name="output")

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.y_pred))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                              global_step=self.global_step_tensor)
            correct_predictions = tf.equal(tf.round(tf.nn.sigmoid(self.y_pred)), tf.round(self.y))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            self.predictions_scores = tf.nn.sigmoid(self.y_pred)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)

import tensorflow as tf
import numpy as np

class WatermarkRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, strength, embed_dim, seed=0):
        self.strength = strength
        self.embed_dim = embed_dim
        self.seed = seed
        self.matrix = None

    def __call__(self, weights):
        self.weights = weights

        # define the watermark
        signature = np.ones((1, self.embed_dim))

        # set a seed
        np.random.seed(self.seed)

        # build the projection matrix for the watermark embedding
        weights_shape = tf.shape(weights)
        mat_rows = np.prod(weights_shape[0:3])
        mat_cols = signature.shape[1]
        self.matrix = np.random.randn(mat_rows, mat_cols)

        # compute cross-entropy loss
        weights_mean = tf.reduce_mean(weights, axis=3)
        weights_flat = tf.reshape(weights_mean, (1, tf.size(weights_mean)))
        proj_matrix = tf.convert_to_tensor(self.matrix, dtype=tf.float32)

        regularized_loss = self.strength * tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(
                tf.sigmoid(tf.matmul(weights_flat, proj_matrix)), signature))

        # apply a penalty to the loss function
        return regularized_loss

    def get_matrix(self):
        return self.matrix

    def get_config(self):
        return {'strength': self.strength}
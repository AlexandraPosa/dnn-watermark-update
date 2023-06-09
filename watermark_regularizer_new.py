import tensorflow as tf

class WatermarkRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, strength):
        self.strength = strength
        self.uses_learning_phase = True

    def __call__(self, weights):
        self.weights = weights

        # define signature
        signature = tf.ones((1, 256))

        # build random matrix
        weights_shape = tf.shape(weights)
        mat_rows = tf.reduce_prod(weights_shape[0:3])
        mat_cols = signature.shape[1]
        matrix = tf.random.normal((mat_rows, mat_cols))

        # compute cross-entropy loss
        weights_mean = tf.reduce_mean(weights, axis=3)
        weights_flat = tf.reshape(weights_mean, (1, tf.size(weights_mean)))

        regularized_loss = self.strength * tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(
                tf.sigmoid(tf.matmul(weights_flat, matrix)), signature))

        return regularized_loss

    def get_config(self):
        return {'strength': self.strength}
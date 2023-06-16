import tensorflow as tf

class WatermarkRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, strength, embed_dim, seed=0):
        self.strength = strength
        self.embed_dim = embed_dim
        self.seed = seed
        self.matrix = None

    def __call__(self, weights):
        self.weights = weights

        # define the watermark
        signature = tf.ones((1, self.embed_dim))

        # set a seed
        tf.random.set_seed(self.seed)

        # build the projection matrix for the watermark embedding
        weights_shape = tf.shape(weights)
        mat_rows = tf.reduce_prod(weights_shape[0:3])
        mat_cols = signature.shape[1]
        self.matrix = tf.random.normal((mat_rows, mat_cols))

        # compute cross-entropy loss
        weights_mean = tf.reduce_mean(weights, axis=3)
        weights_flat = tf.reshape(weights_mean, (1, tf.size(weights_mean)))

        regularized_loss = self.strength * tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(
                tf.sigmoid(tf.matmul(weights_flat, self.matrix)), signature))

        # apply a penalty to the loss function
        return regularized_loss

    def get_matrix(self):
        return self.matrix.numpy()

    def get_config(self):
        return {'strength': self.strength}
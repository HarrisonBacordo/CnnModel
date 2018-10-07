import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm(inputs, training, data_format):
    """

    :param inputs:
    :param training:
    :param data_format:
    :return:
    """
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
    """
    Pads the input independent of its dimensions
    :param inputs:
    :param kernel_size:
    :param data_format:
    :return:
    """
    pad_total = kernel_size - 1
    pad_start = pad_total // 2
    pad_end = pad_total - pad_start
    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_start, pad_end], [pad_start, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_start, pad_end],
                                        [pad_start, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """
    Creates a convolution of fixed padding
    :param inputs:
    :param filters:
    :param kernel_size:
    :param strides:
    :param data_format:
    :return:
    """
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer())


def conv_block(inputs, filters, kernel_size, strides, training, data_format):
    """
    Builds a conv block
    :param inputs:
    :param filters:
    :param kernel_size:
    :param strides:
    :param training:
    :param data_format:
    :return:
    """
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=kernel_size,
                                  strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    return inputs


def res_block(inputs, filters, kernel_size, strides, projection_shortcut, training, data_format):
    """
    Builds a res block
    :param inputs:
    :param filters:
    :param kernel_size:
    :param strides:
    :param projection_shortcut:
    :param training:
    :param data_format:
    :return:
    """
    shortcut = inputs

    if shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(shortcut, training, data_format)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=kernel_size,
                                  strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=kernel_size,
                                  strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def policy_block(inputs, units, training, data_format, ):
    """
    Builds the policy block of the network
    :param inputs:
    :param units:
    :param training:
    :param data_format:
    :return:
    """
    inputs = conv_block(inputs=inputs, filters=2, kernel_size=1, strides=1, training=training,
                        data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    batch_size = inputs.get_shape()[0]
    inputs = tf.reshape(inputs, shape=[batch_size, -1])
    inputs = tf.layers.dense(inputs, units, activation=tf.nn.softmax)
    return inputs


def value_block(inputs, units, training, data_format):
    """
    Builds the value block of the network
    :param inputs:
    :param units:
    :param training:
    :param data_format:
    :return:
    """
    inputs = conv_block(inputs=inputs, filters=1, kernel_size=1, strides=1, training=training,
                        data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    batch_size = inputs.get_shape()[0]
    inputs = tf.reshape(inputs, shape=[batch_size, -1])
    inputs = tf.layers.dense(inputs, units, activation=tf.nn.relu)
    inputs = tf.layers.dense(inputs, 1, activation=tf.nn.tanh)
    return inputs

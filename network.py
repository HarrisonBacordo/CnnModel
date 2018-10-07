import tensorflow as tf
from blocks import conv_block, res_block, conv2d_fixed_padding, value_block, policy_block

NUM_BLOCKS = 18
CONV_FILTERS = 256
CONV_KERNEL = 3
CONV_STRIDES = 1
DATA_FORMAT = 'channels_last'  # TODO MIGHT CAUSE PROBLEMS HERE?
NUM_OUTPUTS = 3
INIT_LEARN_RATE = 0.1
DECAY_STEPS = 100000
DECAY_RATE = 0.1

NUM_EPOCHS = 1
BATCH_SIZE = 1
BOARD_WIDTH = 3
BOARD_HEIGHT = 3
BOARD_SIZE = 3 * 3
STEP_HISTORY = 3
NUM_PLAYERS = 2
CONSTANT_VALUE_INPUT = {
    'MOVE_COUNT': tf.placeholder(dtype=tf.int32)
}
# GAME_PLANES = (STEP_HISTORY * NUM_PLAYERS + len(CONSTANT_VALUE_INPUT))
GAME_PLANES = 1


def projection_shortcut(inputs):
    """

    :param inputs:
    :return:
    """
    return conv2d_fixed_padding(
        inputs=inputs, filters=CONV_FILTERS, kernel_size=1, strides=CONV_STRIDES,
        data_format=DATA_FORMAT)


##################################################################
# ARCHITECTURE
##################################################################
inputs = tf.placeholder(shape=[1, 3, 3, 1], dtype=tf.float32)
training = tf.placeholder(tf.bool)


def res_model():
    """
    Inputs a mini-batch into the residual network model
    :return: the logits for both heads, as well as the most predicted action and its probability
    """

    # Initial convolution block
    output = conv_block(inputs=inputs, filters=CONV_FILTERS, kernel_size=CONV_KERNEL,
                        strides=CONV_STRIDES, training=training, data_format=DATA_FORMAT)
    # Residual blocks
    for _ in range(NUM_BLOCKS):
        output = res_block(inputs=inputs, filters=CONV_FILTERS, kernel_size=CONV_KERNEL,
                           strides=CONV_STRIDES, projection_shortcut=projection_shortcut,
                           training=training, data_format=DATA_FORMAT)

    # Evaluate policy head
    logits = policy_block(output, NUM_OUTPUTS, training, DATA_FORMAT)[0]
    prediction = tf.argmax(logits)
    probability = tf.reduce_max(logits)
    tf.summary.histogram("predictions", logits)

    return logits, prediction, probability


def train_res(logits, labels):
    """
    Implements an optimizer on the residual network model
    :param logits: prediction of the residual network
    :param labels: labels returned by MCTS
    :return: the optimizer and the loss
    """
    learning_rate = tf.train.exponential_decay(INIT_LEARN_RATE, global_step, DECAY_STEPS, DECAY_RATE)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    return optimizer, loss


# Setup
global_step = tf.Variable(0, trainable=False)
saver = tf.train.Saver()

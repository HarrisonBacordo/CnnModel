#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2018 Created by Yiming Peng and Bing Xue
"""

import glob
import numpy as np
import tensorflow as tf
import random
from PIL import Image

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)

strawberry_path = 'data/strawberry/*.jpg'
cherry_path = 'data/cherry/*.jpg'
tomato_path = 'data/tomato/*.jpg'

NUM_OUTPUTS = 3
INIT_LEARN_RATE = 0.0001
BATCH_SIZE = 128

tf.logging.set_verbosity(tf.logging.INFO)


def load_data():
    strawberry_list = glob.glob(strawberry_path)
    cherry_list = glob.glob(cherry_path)
    tomato_list = glob.glob(tomato_path)
    labels = list()
    strawberry_data = np.array([np.array(Image.open(fname).convert("RGB")) for fname in strawberry_list])
    cherry_data = np.array([np.array(Image.open(fname).convert("RGB")) for fname in cherry_list])
    tomato_data = np.array([np.array(Image.open(fname).convert("RGB")) for fname in tomato_list])
    data = np.concatenate((strawberry_data, cherry_data, tomato_data))
    for i in range(3):
        labels += [i] * 1500
    return data.astype(np.float32), np.array(labels)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    inputs = tf.image.resize_images(features["x"], (254, 254))
    if mode == tf.estimator.ModeKeys.TRAIN:
        inputs = tf.random_crop(inputs, [tf.shape(inputs)[0], 227, 227, 3])
        inputs = tf.image.random_flip_left_right(inputs)
        inputs = tf.image.random_flip_up_down(inputs)

    # Convolutional block
    inputs = tf.layers.conv2d(inputs, filters=64, kernel_size=11, strides=4, padding='valid', activation=tf.nn.relu)
    inputs = tf.layers.max_pooling2d(inputs, pool_size=3, strides=2)
    inputs = tf.layers.conv2d(inputs, filters=192, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu)
    inputs = tf.layers.max_pooling2d(inputs, pool_size=3, strides=2)
    inputs = tf.layers.conv2d(inputs, filters=384, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    inputs = tf.layers.conv2d(inputs, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    inputs = tf.layers.conv2d(inputs, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    inputs = tf.layers.max_pooling2d(inputs, pool_size=3, strides=2)

    # fully connected block
    inputs = tf.layers.flatten(inputs)
    inputs = tf.layers.dropout(inputs, rate=0.5)
    inputs = tf.layers.dense(inputs, units=4096, activation=tf.nn.relu)
    inputs = tf.layers.dropout(inputs, rate=0.5)
    inputs = tf.layers.dense(inputs, units=4096, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs, NUM_OUTPUTS)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=INIT_LEARN_RATE)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and validation data
    train_data = np.load("data/train/train_data.npy")
    train_labels = np.load("data/train/train_labels.npy")
    validation_data = np.load("data/train/validation_data.npy")
    validation_labels = np.load("data/train/validation_labels.npy")

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="chkpts/81er")

    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    # Train the model
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": train_data},
    #     y=train_labels,
    #     batch_size=BATCH_SIZE,
    #     num_epochs=None,
    #     shuffle=True)
    # classifier.train(
    #     input_fn=train_input_fn,
    #     steps=4000,
    #     hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": validation_data},
        y=validation_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()

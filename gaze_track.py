#!/usr/bin/python3

import argparse
import logging
import os
import timeit

import numpy as np
import tensorflow as tf


def conv_layer(input_var, chan_out, name, size=5):
    ''' Create a convolutional layer '''
    with tf.name_scope(name):
        chan_in = int(input_var.get_shape()[3])
        w = tf.Variable(tf.truncated_normal([size, size, chan_in, chan_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[chan_out]), name="B")

        return tf.nn.conv2d(input_var, w, [1, 1, 1, 1], padding="VALID") + b

def fc_layer(input_var, chan_out, name):
    ''' Create a fully connected layer '''
    with tf.name_scope(name):
        chan_in = int(input_var.get_shape()[1])
        w = tf.Variable(tf.random_normal([chan_in, chan_out]), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[chan_out]), name="B")

        return tf.matmul(input_var, w) + b

def max_pool(input_var):
    ''' Do max pool '''
    return tf.nn.max_pool(input_var, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding="VALID")

def lrn(input_var):
    ''' Local response normalization '''
    return tf.nn.local_response_normalization(input_var)

def relu(input_var):
    ''' Relu '''
    return tf.nn.relu(input_var)

class GazeTrack():

    def __init__(self):
        self.sess = tf.Session()
        self.batch_size = 256
        self.location = './test/'
        self.build_graph()

    def build_graph(self):
        ''' Build the graph '''
        with self.sess:
            # Create placeholder
            self.left_eye = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                           name="left_eye")
            self.righ_eye = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                           name="right_eye")
            self.face = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                       name="face")
            self.face_mask = tf.placeholder(tf.float32, [None, 25, 25],
                                           name="face_mask")

            self.y = tf.placeholder(tf.float32, [None, 2], name="target")

            self.learning_rate = tf.placeholder(tf.float32, [],
                                                name="learning_rate")
            self.global_step = tf.Variable(initial_value=0,
                                           trainable=False, name='global_step')

            left_conv = self.left_eye
            righ_conv = self.righ_eye
            face_conv = self.face

            # Build convolutional layers according to parameters
            conv_param = [32, 64, 128]
            conv_size = [3, 4, 4]
            conv_pool = [True, True, True]
            for i in range(len(conv_param)):
                # Conv layer
                left_conv = conv_layer(left_conv, conv_param[i],
                                       "left_conv_" + str(i), size=conv_size[i])
                righ_conv = conv_layer(righ_conv, conv_param[i],
                                       "right_conv_" + str(i), size=conv_size[i])
                face_conv = conv_layer(face_conv, conv_param[i],
                                       "face_conv_" + str(i), size=conv_size[i])
                # Activation layer
                left_conv = lrn(left_conv)
                righ_conv = lrn(righ_conv)
                face_conv = lrn(face_conv)

                if conv_pool[i]:
                    # Pool
                    left_conv = max_pool(left_conv)
                    righ_conv = max_pool(righ_conv)
                    face_conv = max_pool(face_conv)

            # Reshape convolutional layer output
            shape = left_conv.get_shape()
            conv_out = int(shape[1] * shape[2] * shape[3])
            left_conv_out = tf.reshape(left_conv, [-1, conv_out])
            righ_conv_out = tf.reshape(righ_conv, [-1, conv_out])
            face_conv_out = tf.reshape(face_conv, [-1, conv_out])

            # Build fully connected layer
            fc = tf.concat([left_conv_out, righ_conv_out, face_conv_out,
                            tf.reshape(self.face_mask, [-1, 25*25])], 1)
            fc = fc_layer(fc, 128, "fc_0")
            fc = relu(fc)
            self.out = fc_layer(fc, 2, "out")

            # Validate solution
            with tf.name_scope("validation"):
                ydis = tf.sqrt(tf.reduce_sum(tf.squared_difference(self.out,
                                                                   self.y), 1))
                # Record loss function for training data
                self.loss = tf.reduce_mean(ydis, name='loss')
                self.loss_sum = tf.summary.scalar("loss", self.loss)
                # Record loss function for validation data
                self.loss_vali = tf.reduce_mean(ydis)
                self.loss_vali_sum = tf.summary.scalar("loss_vali", self.loss_vali)
                # Return a sample solution
                self.out_sample = tf.gather(self.out, 0)
                self.ydis_sample = tf.gather(ydis, 0)

            # Train
            with tf.name_scope("train"):
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).\
                                      minimize(ydis, global_step=self.global_step)

            # Prepare for output
            self.saver = tf.train.Saver()
            self.writer = tf.summary.FileWriter(self.location)

    def load_model(self):
        ''' Load existing model or initializing new model '''
        if self.location is not None:
            try:
                path = tf.train.latest_checkpoint(checkpoint_dir=self.location)
                self.saver.restore(self.sess, save_path=path)
                logging.info("Restoring from ", path)
                return True
            except:
                pass

        logging.info("Initializing variables")
        self.sess.run(tf.global_variables_initializer())
        return False

    def train(self, learning_rate, steps, data, vali_data):
        ''' Train model '''

        print_it = 10
        save_it = 1000
        vali_it = 100

        # Try to load old model
        loaded = self.load_model()

        if not loaded:
            self.writer.add_graph(self.sess.graph)

        it = self.sess.run(self.global_step)
        logging.info("Start training from %d it\n" %it)

        start_t = timeit.default_timer()
        last_t = start_t
        last_it = it
        start_it = it

        for i in range(it, steps):
            it = i + 1
            indx = np.random.choice(len(data), self.batch_size)
            b_left, b_right, b_face, b_mask, b_y = data['left'][indx], data['left'][indx], \
                                                   data['face'][indx], data['mask'][indx], \
                                                   data['y'][indx]

            _, l, s = self.sess.run([self.train_step, self.loss, self.loss_sum], \
                                    {self.left_eye: b_left, self.righ_eye: b_right, \
                                     self.face: b_face, self.face_mask: b_mask, \
                                     self.y: b_y, self.learning_rate: learning_rate})
            self.writer.add_summary(s, it)

            # Print loss
            if it % print_it == 0 or it == 1:

                logging.info((str(it) + ': ').ljust(7) + str(l))

                # Validate model
                if it % vali_it == 0 or it == 1:
                    end_t = timeit.default_timer()

                    l_vali, s = self.validate(vali_data)
                    self.writer.add_summary(s, it)

                    logging.info((str(it) + ': ').ljust(7) + 'validation: ' + str(l_vali))
                    logging.info((str(it) + ': ').ljust(7) + '%.3fs from last validation with average %.3fs/it' \
                                 %(end_t - last_t, (end_t - last_t)/(it - last_it)))

                    last_t = timeit.default_timer()
                    last_it = it

            # Save model
            if it % save_it == 0:
                save_path = self.saver.save(self.sess, self.location + 'model', \
                                            global_step=self.global_step)
                logging.info((str(it) + ': ').ljust(7) + 'Model saved')

        if it % save_it != 0:
            save_path = self.saver.save(self.sess, self.location + 'model', \
                                        global_step=self.global_step)
            logging.info((str(it) + ': ').ljust(7) + 'Model saved')

        end_t = timeit.default_timer()
        logging.info('\n%.3fs with %d it' % (end_t - start_t, steps))
        logging.info('Average time: %f' % ((end_t - start_t) / (steps - start_it)))

    def validate(self, data):
        ''' Validate the performance of model '''
        indx = np.random.choice(len(data), self.batch_size)

        b_left, b_right, b_face, b_mask, b_y = data['left'][indx], data['left'][indx], \
                                               data['face'][indx], data['mask'][indx], \
                                               data['y'][indx]
        loss, s = self.sess.run([self.loss_vali, self.loss_vali_sum], \
                                {self.left_eye: b_left, self.righ_eye: b_right, \
                                 self.face: b_face, self.face_mask: b_mask, \
                                 self.y: b_y})

        return loss, s


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('learning_rate', type=float)
    parser.add_argument('steps', type=int)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', \
                        level=logging.INFO)

    model = GazeTrack()

    data = {}
    vali_data = {}
    npzfile = np.load("train_and_val.npz")
    data['left'] = npzfile["train_eye_left"] / 225.       # 64x64x3
    data['right'] = npzfile["train_eye_right"] / 225.     # 64x64x3
    data['face'] = npzfile["train_face"] / 225.           # 64x64x3
    data['mask'] = npzfile["train_face_mask"]             # 25x25
    data['y'] = npzfile["train_y"]

    vali_data['left'] = npzfile["val_eye_left"] / 225.
    vali_data['right'] = npzfile["val_eye_right"] / 225.
    vali_data['face'] = npzfile["val_face"] / 225.
    vali_data['mask'] = npzfile["val_face_mask"]
    vali_data['y'] = npzfile["val_y"]

    npzfile = None

    model.train(args.learning_rate, args.steps, data, vali_data)

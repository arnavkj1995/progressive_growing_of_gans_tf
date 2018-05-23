import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
from scipy.misc import imread, imsave
# import logging
import socket

import progressive_model

parser = argparse.ArgumentParser()
# parser.add_argument('--image', default='', type=str,
#                     help='The filename of image to be completed.')
# parser.add_argument('--mask', default='', type=str,
#                     help='The filename of mask, value 255 indicates mask.')
# parser.add_argument('--output', default='output.png', type=str,
#                     help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    config = ng.Config('progressive_gan.yml')
    if config.GPU_ID != -1:
        ng.set_gpus(config.GPU_ID)
    else:
        ng.get_gpus(config.NUM_GPUS)
    np.random.seed(config.RANDOM_SEED)

    # ng.get_gpus(1)
    args = parser.parse_args()

    model = progressive_model.ProgressiveGAN(1024, config)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        z = tf.placeholfer(tf.foat32, [config.BATCH_SIZE, 1, 1, 512], -1, 1, name='z')
        mask = tf.placeholder(tf.float32, [config.BATCH_SIZE, config.CURRENT_RESOLUTION, config.CURRENT_RESOLUTION, 3], name='mask')
        orig = tf.placeholder(tf.float32, [config.BATCH_SIZE, config.CURRENT_RESOLUTION, config.CURRENT_RESOLUTION, 3], name='real_images')

        # orig_ = orig * mask

        outputs, grads = model.build_graph_with_opt(z, mask, orig, config)

        # output = (output + 1.) * 127.5
        # output = tf.reverse(output, [-1])
        # output = tf.saturate_cast(output, tf.uint8)

        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))

        ### Implement iterative opt here
        z_ = np.random.uniform(-1, 1, ([config.BATCH_SIZE, 1, 1, 512]))
        orig_ = np.random.uniform(-1, 1, [config.BATCH_SIZE, config.CURRENT_RESOLUTION, config.CURRENT_RESOLUTION, 3])
        mask_ = np.random.randint( 0, 2,  [config.BATCH_SIZE, config.CURRENT_RESOLUTION, config.CURRENT_RESOLUTION, 3])

        imgs_out = sess.run(outputs, feed_dict={z:z_, orig:orig_, mask:mask_})

        imsave('sample.png', imgs_out[0])
        # print('Model loaded.')
        # result = sess.run(output)
        # cv2.imwrite(args.output, result[0][:, :, ::-1])

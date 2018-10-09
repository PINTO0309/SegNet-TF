import tensorflow as tf
import numpy as np
from layers_object_infer import conv_layer, up_sampling, max_pool, initialization, variable_with_weight_decay

class SegNet:
    def __init__(self):

        self.vgg_param_dict = np.load("vgg16.npy", encoding='latin1').item()
        self.inputs_pl = tf.placeholder(tf.float32, [None, 240, 320, 3], name="input")

        self.norm1 = tf.nn.lrn(self.inputs_pl, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')
        # first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
        self.conv1_1 = conv_layer(self.norm1, "conv1_1", [3, 3, 3, 64], False, True, self.vgg_param_dict)
        self.conv1_2 = conv_layer(self.conv1_1, "conv1_2", [3, 3, 64, 64], False, True, self.vgg_param_dict)
        self.pool1, self.pool1_index, self.shape_1 = max_pool(self.conv1_2, 'pool1')

        # Second box of convolution layer(4)
        self.conv2_1 = conv_layer(self.pool1, "conv2_1", [3, 3, 64, 128], False, True, self.vgg_param_dict)
        self.conv2_2 = conv_layer(self.conv2_1, "conv2_2", [3, 3, 128, 128], False, True, self.vgg_param_dict)
        self.pool2, self.pool2_index, self.shape_2 = max_pool(self.conv2_2, 'pool2')

        # Third box of convolution layer(7)
        self.conv3_1 = conv_layer(self.pool2, "conv3_1", [3, 3, 128, 256], False, True, self.vgg_param_dict)
        self.conv3_2 = conv_layer(self.conv3_1, "conv3_2", [3, 3, 256, 256], False, True, self.vgg_param_dict)
        self.conv3_3 = conv_layer(self.conv3_2, "conv3_3", [3, 3, 256, 256], False, True, self.vgg_param_dict)
        self.pool3, self.pool3_index, self.shape_3 = max_pool(self.conv3_3, 'pool3')

        # Fourth box of convolution layer(10)
        self.conv4_1 = conv_layer(self.pool3, "conv4_1", [3, 3, 256, 512], False, True, self.vgg_param_dict)
        self.conv4_2 = conv_layer(self.conv4_1, "conv4_2", [3, 3, 512, 512], False, True, self.vgg_param_dict)
        self.conv4_3 = conv_layer(self.conv4_2, "conv4_3", [3, 3, 512, 512], False, True, self.vgg_param_dict)
        self.pool4, self.pool4_index, self.shape_4 = max_pool(self.conv4_3, 'pool4')

        # Fifth box of convolution layers(13)
        self.conv5_1 = conv_layer(self.pool4, "conv5_1", [3, 3, 512, 512], False, True, self.vgg_param_dict)
        self.conv5_2 = conv_layer(self.conv5_1, "conv5_2", [3, 3, 512, 512], False, True, self.vgg_param_dict)
        self.conv5_3 = conv_layer(self.conv5_2, "conv5_3", [3, 3, 512, 512], False, True, self.vgg_param_dict)
        self.pool5, self.pool5_index, self.shape_5 = max_pool(self.conv5_3, 'pool5')

        # ---------------------So Now the encoder process has been Finished--------------------------------------#
        # ------------------Then Let's start Decoder Process-----------------------------------------------------#

        # First box of deconvolution layers(3)
        self.deconv5_1 = up_sampling(self.pool5, self.pool5_index, self.shape_5, 1, name="unpool_5")
        self.deconv5_2 = conv_layer(self.deconv5_1, "deconv5_2", [3, 3, 512, 512], False)
        self.deconv5_3 = conv_layer(self.deconv5_2, "deconv5_3", [3, 3, 512, 512], False)
        self.deconv5_4 = conv_layer(self.deconv5_3, "deconv5_4", [3, 3, 512, 512], False)
        # Second box of deconvolution layers(6)
        self.deconv4_1 = up_sampling(self.deconv5_4, self.pool4_index, self.shape_4, 1, name="unpool_4")
        self.deconv4_2 = conv_layer(self.deconv4_1, "deconv4_2", [3, 3, 512, 512], False)
        self.deconv4_3 = conv_layer(self.deconv4_2, "deconv4_3", [3, 3, 512, 512], False)
        self.deconv4_4 = conv_layer(self.deconv4_3, "deconv4_4", [3, 3, 512, 256], False)
        # Third box of deconvolution layers(9)
        self.deconv3_1 = up_sampling(self.deconv4_4, self.pool3_index, self.shape_3, 1, name="unpool_3")
        self.deconv3_2 = conv_layer(self.deconv3_1, "deconv3_2", [3, 3, 256, 256], False)
        self.deconv3_3 = conv_layer(self.deconv3_2, "deconv3_3", [3, 3, 256, 256], False)
        self.deconv3_4 = conv_layer(self.deconv3_3, "deconv3_4", [3, 3, 256, 128], False)
        # Fourth box of deconvolution layers(11)
        self.deconv2_1 = up_sampling(self.deconv3_4, self.pool2_index, self.shape_2, 1, name="unpool_2")
        self.deconv2_2 = conv_layer(self.deconv2_1, "deconv2_2", [3, 3, 128, 128], False)
        self.deconv2_3 = conv_layer(self.deconv2_2, "deconv2_3", [3, 3, 128, 64], False)
        # Fifth box of deconvolution layers(13)
        self.deconv1_1 = up_sampling(self.deconv2_3, self.pool1_index, self.shape_1, 1, name="unpool_1")
        self.deconv1_2 = conv_layer(self.deconv1_1, "deconv1_2", [3, 3, 64, 64], False)
        self.deconv1_3 = conv_layer(self.deconv1_2, "deconv1_3", [3, 3, 64, 64], False)

        with tf.variable_scope('conv_classifier') as scope:
            self.kernel = variable_with_weight_decay('weights', initializer=initialization(1, 64), shape=[1, 1, 64, 12], wd=False)
            self.conv = tf.nn.conv2d(self.deconv1_3, self.kernel, [1, 1, 1, 1], padding='SAME')
            self.biases = variable_with_weight_decay('biases', tf.constant_initializer(0.0), shape=[12], wd=False)
            self.logits = tf.nn.bias_add(self.conv, self.biases, name="output")

def main():

    graph = tf.Graph()
    with graph.as_default():
        model = SegNet()

        saver = tf.train.Saver(tf.global_variables())
        sess  = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print("in=", model.inputs_pl.name)
        print("on=", model.logits.name)

        saver.restore(sess, './ckpt/deploy.ckpt')
        saver.save(sess, './ckpt/deployfinal.ckpt')

        graphdef = graph.as_graph_def()
        tf.train.write_graph(graphdef, './ckpt', 'deployfinal.pbtxt', as_text=True)

if __name__ == '__main__':
    main()



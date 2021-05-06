from tensorflow.contrib import slim;
import tensorflow as tf;


class abstract_unpool:
    def get_slime_arg_scope(self, is_training, **kwargs):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        return slim.arg_scope([slim.conv2d],
                              activation_fn=tf.nn.relu,
                              normalizer_fn=slim.batch_norm,
                              normalizer_params=batch_norm_params,
                              weights_regularizer=slim.l2_regularizer(kwargs["weight_decay"]),
                              );
    def model(this, inputs, is_training, num_outputs=[None, 128, 64, 32], **kwargs):
        pass;

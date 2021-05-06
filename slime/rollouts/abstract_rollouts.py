# TODO [V1] unify libeast and libpixellink
import tensorflow as tf;

class abstract_rollout:
    @staticmethod
    def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
        '''
        image normalization
        :param images:
        :param means:
        :return:
        '''
        num_channels = images.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')
        channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=3, values=channels)
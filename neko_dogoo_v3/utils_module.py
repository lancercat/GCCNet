import tensorflow as tf;
class dogoo_ops:
    @staticmethod
    def softmax(target, axis, name=None):
        with tf.name_scope(name, 'softmax', values=[target]):
            max_axis = tf.reduce_max(target, axis, keepdims=True)
            target_exp = tf.exp(target - max_axis)
            normalize = tf.reduce_sum(target_exp, axis, keepdims=True)
            softmax = target_exp / normalize
            return softmax

    @staticmethod
    def unpool(inputs):
        return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])


    '''
    filters: 5d tensor:N,H,W,IC,OC
    feature:4d tensor: N,H,W,IC
    produces: 4d tensor: N,H,W, OC
    '''
    @staticmethod
    def batch_wise_conv2d(features,kernels,**args):
        batch_size=tf.shape(features)[0];
        kernel_arr=tf.TensorArray(kernels.dtype,batch_size);
        feature_arr=tf.TensorArray(features.dtype,batch_size);
        res_arr=tf.TensorArray(features.dtype,batch_size);

        kernel_arr=kernel_arr.unstack(kernels);
        feature_arr=feature_arr.unstack(tf.expand_dims(feature_arr,axis=1));
        def loop_body(i,result_arr):
            f=feature_arr.read(i);
            k=kernel_arr.read(i);
            result_arr= result_arr.write(i,tf.nn.conv2d(f,k,**args));
            return i+1,result_arr;
        def cond(i):
            return i<batch_size;
        i =0;
        i,res_arr=tf.while_loop(cond,loop_body,[i,res_arr]);
        return tf.squeeze(res_arr.stack());

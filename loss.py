
import tensorflow as tf
import math

def build_adacos_fn(num_classes):
    pi =  tf.constant(3.14159265358979323846)
    theta_zero = pi/4
    m = 0.5

    def adacos_fn(labels, logits, is_clean_input, loss_weight_input):
        adacos_s = tf.math.sqrt(2.0) * tf.math.log(tf.cast(num_classes - 1,tf.float32))
        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=num_classes)
        theta = tf.math.acos(tf.clip_by_value(logits, -1.0 + 1e-7, 1.0 - 1e-7))

        B_avg =tf.where(mask==1,tf.zeros_like(logits), tf.math.exp(adacos_s * logits))
        B_avg = tf.reduce_mean(tf.reduce_sum(B_avg, axis=1), name='B_avg')
        B_avg = tf.stop_gradient(B_avg)
        theta_class = tf.gather_nd(theta, tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1), name='theta_class')
        theta_med = tfp.stats.percentile(theta_class, q=50)
        theta_med = tf.stop_gradient(theta_med)

        adacos_s=(tf.math.log(B_avg) / tf.cos(tf.minimum(self.theta_zero, theta_med)))
        output = tf.multiply(self.adacos_s, logits, name='adacos_logits')        
        cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
        isCleanValues = tf.cast(tf.gather_nd(tf.constant([[NOT_CLEAN_WEIGHT],[1.0]]), tf.stack([cleans,tf.tile(tf.constant([0]),tf.shape(cleans))], axis=1)), tf.float32)
        weightValues = tf.cast(tf.gather_nd(lossWeight, tf.stack([tf.range(BATCH_SIZE_PER_TPU),labels], axis=1)),tf.float32)
        loss = cce(labels, output, sample_weight = tf.multiply(isCleanValues, weightValues))
        return loss

    return adacos_fn
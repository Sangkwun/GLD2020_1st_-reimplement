import tensorflow as tf
from layer import ArcMarginProduct_v2
from loss import build_adacos_fn

MODEL_MAP = {
    'efficient-b5': tf.keras.applications.EfficientNetB5,
}

def build_model(model_name, input_size, num_classes):

    image_input = tf.keras.layers.Input(
        shape=[*input_size, 3], dtype='float32', name='image'
    )
    is_clean_input = tf.keras.layers.Input(
        shape=[1], dtype='int64', name='is_clean'
    )
    loss_weight_input = tf.keras.layers.Input(
        shape=[1], dtype='float32', name='loss_weight'
    )
    base_model = MODEL_MAP[model_name](
        input_tensor=image_input,
        include_top=False,
        pooling='avg'
    )

    landmark_id = tf.keras.layers.Input(shape=[num_classes], dtype='int64', name='landmark_id')

    x = base_model(image_input)
    x = tf.keras.layers.Dense(521, activation='swish')(x)
    x = ArcMarginProduct_v2(num_classes)(x)

    adacos_fn = build_adacos_fn(num_classes=num_classes)
    adacos_loss = tf.keras.layers.Lambda(adacos_fn, output_shape=(1,), name='adacos')(
        [landmark_id, x, is_clean_input, loss_weight_input]
    )

    model_inputs = [
        image_input, is_clean_input, loss_weight_input
    ]
    model = tf.keras.Model(inputs=model_inputs, outputs=adacos_loss)
    return model
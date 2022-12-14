# bunch of imports
import tensorflow as tf
from tensorflow.keras import layers, Model
from scipy.ndimage import affine_transform, rotate
from skimage.transform import resize
import numpy as np
import random
import os
from dipy.io.image import save_nifti, load_nifti
import lattice_filter_op_loader

custom_module = lattice_filter_op_loader.module

# setting gpu memory growth
# not important but was needed for multi-GPU runs in some cases
def set_GPU():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # This is telling the default strategy for multi-GPUs
    strategy = tf.distribute.MirroredStrategy()
    return strategy

# prepare image for input
def __prepare_img(file_path):
    image_data, affine = load_nifti(file_path)
    shape = image_data.shape
    image_data = np.squeeze(image_data)
    image_data = __normalize(image_data)
    image_data, new_affine = __transform_img(image_data, affine, (128, 128, 128))
    image_data = np.reshape(image_data, (128, 128, 128, 1))

    return image_data, new_affine, shape


# normalization function
def __normalize(input_image):
    """Internal function for normalizing the images"""
    input_image = (input_image-np.min(input_image)) / (np.max(input_image)-np.min(input_image))
    return input_image

# transform function
# to transfrom all images to the same coordinate space
def __transform_img(input_image, affine, new_shape):
    """Internal function for transforming the images to the same coordinate space"""
    new_affine = affine.copy()
    shape = input_image.shape
    new_affine[:3, 3] += np.array([128, 128, 128])
    inv_affine = np.linalg.inv(new_affine)
    new_img = affine_transform(input_image, inv_affine, output_shape=(256, 256, 256))
    
    final_img = resize(new_img, new_shape)
    return final_img, new_affine

# recover the image back to its original space
def __recover(input_image, affine, new_shape):
    """Internal function for recovering the images to its original space"""
    new_image = resize(input_image, (256, 256, 256))
    new_image = affine_transform(new_image, affine, output_shape=new_shape)
    new_affine = affine.copy()
    new_affine[:3, 3] -= np.array([128, 128, 128])
    return new_image, new_affine

def __largest(image):
    from scipy.ndimage import label
    new_cc_mask = np.zeros(image.shape)
    labels, numL = label(image)
    volumes = [len(labels[np.where(labels == l_idx+1)])
               for l_idx in np.arange(numL)]
    biggest_vol = np.arange(numL)[np.where(volumes == np.max(volumes))] + 1
    new_cc_mask[np.where(labels == biggest_vol)] = 1

    return new_cc_mask

# generate training data
def __data_gen(file_list, label_list, training=False, model_type='evnet'):
    """ Data generator.
    Parameters
    ----------
    file_list : list
        The list of absolute file paths for the images.

    label_list : list
        The list of absolute file paths for the labels.
        Note that file_list and label_list should have the same length.

    training : bool
        Whether the data is for training or testing
        If True, random augmentation will be performed on the files

    Returns
    --------
    denoised array : ndarray
        The 4D denoised DWI data.
    """
    
    random.shuffle(file_list)
    for file_name, label_name in zip(file_list, label_list):
        image_data, affine = load_nifti(file_name)
        image_data = __normalize(image_data)
        label_data, _ = load_nifti(label_name)

        if training:
            # random augmentations
            r_float = np.random.uniform()
            # scaling
            if r_float < 0.125:
                image_data = np.clip(image_data + np.random.uniform(0, 0.1, np.shape(image_data)), 0, 1)
            # shifting
            elif r_float < 0.25:
                image_data = np.clip(image_data * np.random.uniform(0.9, 1.1, np.shape(image_data)), 0, 1)
            # rotation
            elif r_float < 0.375:
                r_float2 = np.random.uniform()
                r_float3 = np.random.uniform()
                r_float4 = np.random.uniform()
                if r_float2 < 0.33:
                    if r_float3 < 0.5:
                        image_data = rotate(image_data, 15.0*r_float4, (0, 1), reshape=False)
                        label_data = rotate(label_data, 15.0*r_float4, (0, 1), reshape=False)
                    else:
                        image_data = rotate(image_data, 360 - 15.0*r_float4, (0, 1), reshape=False)
                        label_data = rotate(label_data, 360 - 15.0*r_float4, (0, 1), reshape=False)
                if r_float2 < 0.33:
                    if r_float3 < 0.5:
                        image_data = rotate(image_data, 15.0*r_float4, (0, 2), reshape=False)
                        label_data = rotate(label_data, 15.0*r_float4, (0, 2), reshape=False)
                    else:
                        image_data = rotate(image_data, 360 - 15.0*r_float4, (0, 2), reshape=False)
                        label_data = rotate(label_data, 360 - 15.0*r_float4, (0, 2), reshape=False)
                else:
                    if r_float3 < 0.5:
                        image_data = rotate(image_data, 15.0*r_float4, (1, 2), reshape=False)
                        label_data = rotate(label_data, 15.0*r_float4, (1, 2), reshape=False)
                    else:
                        image_data = rotate(image_data, 360 - 15.0*r_float4, (1, 2), reshape=False)
                        label_data = rotate(label_data, 360 - 15.0*r_float4, (1, 2), reshape=False)
            # trnaslation
            elif r_float < 0.5:
                r_float2 = int(np.round(np.random.uniform() * 10))
                r_float3 = int(np.round(np.random.uniform() * 10))
                r_float4 = int(np.round(np.random.uniform() * 10))
                r_float5 = np.random.uniform()
                if r_float5 < 0.125:
                    image_data = np.pad(image_data[r_float2:, r_float3:, r_float4:], ((0,r_float2), (0,r_float3), (0,r_float4)), mode='constant')
                    label_data = np.pad(label_data[r_float2:, r_float3:, r_float4:], ((0,r_float2), (0,r_float3), (0,r_float4)), mode='constant')
                if r_float5 < 0.25:
                    image_data = np.pad(image_data[r_float2:, r_float3:, :-r_float4], ((0,r_float2), (0,r_float3), (r_float4, 0)), mode='constant')
                    label_data = np.pad(label_data[r_float2:, r_float3:, :-r_float4], ((0,r_float2), (0,r_float3), (r_float4, 0)), mode='constant')
                if r_float5 < 0.375:
                    image_data = np.pad(image_data[r_float2:, :-r_float3, r_float4:], ((0,r_float2), (r_float3, 0), (0,r_float4)), mode='constant')
                    label_data = np.pad(label_data[r_float2:, :-r_float3, r_float4:], ((0,r_float2), (r_float3, 0), (0,r_float4)), mode='constant')
                if r_float5 < 0.5:
                    image_data = np.pad(image_data[r_float2:, :-r_float3, :-r_float4], ((0,r_float2), (r_float3, 0), (r_float4, 0)), mode='constant')
                    label_data = np.pad(label_data[r_float2:, :-r_float3, :-r_float4], ((0,r_float2), (r_float3, 0), (r_float4, 0)), mode='constant')
                if r_float5 < 0.625:
                    image_data = np.pad(image_data[:-r_float2, r_float3:, r_float4:], ((r_float2, 0), (0,r_float3), (0,r_float4)), mode='constant')
                    label_data = np.pad(label_data[:-r_float2, r_float3:, r_float4:], ((r_float2, 0), (0,r_float3), (0,r_float4)), mode='constant')
                if r_float5 < 0.75:
                    image_data = np.pad(image_data[:-r_float2, r_float3:, :-r_float4], ((r_float2, 0), (0,r_float3), (r_float4, 0)), mode='constant')
                    label_data = np.pad(label_data[:-r_float2, r_float3:, :-r_float4], ((r_float2, 0), (0,r_float3), (r_float4, 0)), mode='constant')
                if r_float5 < 0.875:
                    image_data = np.pad(image_data[:-r_float2, :-r_float3, r_float4:], ((r_float2, 0), (r_float3, 0), (0,r_float4)), mode='constant')
                    label_data = np.pad(label_data[:-r_float2, :-r_float3, r_float4:], ((r_float2, 0), (r_float3, 0), (0,r_float4)), mode='constant')
                else:
                    image_data = np.pad(image_data[:-r_float2, :-r_float3, :-r_float4], ((r_float2, 0), (r_float3, 0), (r_float4, 0)), mode='constant')
                    label_data = np.pad(label_data[:-r_float2, :-r_float3, :-r_float4], ((r_float2, 0), (r_float3, 0), (r_float4, 0)), mode='constant')
        image_data, _ = __transform_img(image_data, affine, (128, 128, 128))
        image_data = np.expand_dims(image_data, -1)
        label_data, _ = __transform_img(label_data, affine, (128, 128, 128))
        if model_type == 'vnet':
            yield (image_data, label_data)
        else:
            yield ({"input_1": image_data, "input_2": resize(image_data, (64, 64, 64, 1)), "input_3": resize(image_data, (32, 32, 32, 1)), "input_4": resize(image_data, (16, 16, 16, 1)), "input_5": resize(image_data, (8, 8, 8, 1))}, label_data)


class dice_coef(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, y_pred.dtype)
        pred_ssum = tf.math.reduce_sum(tf.math.square(y_pred), (1, 2, 3))
        target_ssum = tf.math.reduce_sum(tf.math.square(y_true), (1, 2, 3))
        mul_sum = tf.math.reduce_sum(tf.math.multiply_no_nan(y_pred, y_true), (1, 2, 3))
        
        res = tf.math.divide_no_nan(-2 * mul_sum, tf.math.add(pred_ssum, target_ssum))
        return 1 + res



def create_dataset(files, labels, training='False', batch_size=2):
    """ Function to create a tensorflow dataset object.
    Parameters
    ----------
    file_list : list
        The list of absolute file paths for the images.

    label_list : list
        The list of absolute file paths for the labels.
        Note that file_list and label_list should have the same length.

    training : bool
        Whether the data is for training or testing
        If True, random augmentation will be performed on the files
    
    model_type : str
        The type of model. Can be 'vnet', 'evnet' or 'evcnet'
    
    batch_size : int
        The size of the mini batch. Consider lowering this value if
        you get an out of memory error

    Returns
    --------
    denoised array : ndarray
        The 4D denoised DWI data.
    """
    
    output_types=({"input_1":tf.float32, "input_2":tf.float32, "input_3":tf.float32, "input_4":tf.float32, "input_5":tf.float32}, tf.float32)
    output_shapes=({"input_1":(128, 128, 128, 1), "input_2":(64, 64, 64, 1), "input_3":(32, 32, 32, 1), "input_4":(16, 16, 16, 1), "input_5":(8, 8, 8, 1)}, (128, 128, 128))
    
    
    ds = tf.data.Dataset.from_generator(
    lambda: __data_gen(files, labels, training=training, model_type=model_type),
    output_types=output_types,
    output_shapes=output_shapes
    )

    ds = ds.batch(batch_size)

    return ds

   

class dice_coef_real(tf.keras.metrics.Metric):

    def __init__(self, name='dice_coef_real', **kwargs):
        super(dice_coef_real, self).__init__(name=name, **kwargs)
        self.dice_v = self.add_weight(name='dice_real', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float32)
        after_mask = y_pred[..., 0]
        y_true = tf.cast(y_true, y_pred.dtype)
        pred_ssum = tf.math.reduce_sum(tf.math.square(after_mask), (1, 2, 3))
        target_ssum = tf.math.reduce_sum(tf.math.square(y_true), (1, 2, 3))
        mul_sum = tf.math.reduce_sum(tf.math.multiply_no_nan(after_mask, y_true), (1, 2, 3))
        res = 1 + tf.math.divide_no_nan(-2 * mul_sum, tf.math.add(pred_ssum, target_ssum))

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, res.dtype)
            sample_weight = tf.broadcast_to(sample_weight, res.shape)
            res = tf.multiply(res, sample_weight)

        self.dice_v.assign(tf.reduce_sum(res))

    def result(self):
        return self.dice_v

class dice_coef_orig(tf.keras.metrics.Metric):

    def __init__(self, name='dice_coef_orig', **kwargs):
        super(dice_coef_orig, self).__init__(name=name, **kwargs)
        self.dice_v = self.add_weight(name='dice_orig', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float32)
        before_mask = y_pred[..., 2]
        y_true = tf.cast(y_true, y_pred.dtype)
        pred_ssum = tf.math.reduce_sum(tf.math.square(before_mask), (1, 2, 3))
        target_ssum = tf.math.reduce_sum(tf.math.square(y_true), (1, 2, 3))
        mul_sum = tf.math.reduce_sum(tf.math.multiply_no_nan(before_mask, y_true), (1, 2, 3))
        res = tf.math.divide_no_nan(-2 * mul_sum, tf.math.add(pred_ssum, target_ssum))

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, res.dtype)
            sample_weight = tf.broadcast_to(sample_weight, res.shape)
            res = tf.multiply(res, sample_weight)

        self.dice_v.assign(tf.reduce_sum(res))

    def result(self):
        return self.dice_v

class dice_coef_comp(tf.keras.metrics.Metric):

    def __init__(self, name='dice_coef_comp', **kwargs):
        super(dice_coef_comp, self).__init__(name=name, **kwargs)
        self.dice_v = self.add_weight(name='dice_comp', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float32)
        before_mask = y_pred[..., 2]
        after_mask = y_pred[..., 0]
        y_true = tf.cast(y_true, y_pred.dtype)
        pred_ssum = tf.math.reduce_sum(tf.math.square(before_mask), (1, 2, 3))
        target_ssum = tf.math.reduce_sum(tf.math.square(after_mask), (1, 2, 3))
        mul_sum = tf.math.reduce_sum(tf.math.multiply_no_nan(before_mask, after_mask), (1, 2, 3))
        res = tf.math.divide_no_nan(2 * mul_sum, tf.math.add(pred_ssum, target_ssum))

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, res.dtype)
            sample_weight = tf.broadcast_to(sample_weight, res.shape)
            res = tf.multiply(res, sample_weight)

        self.dice_v.assign(tf.reduce_sum(res))

    def result(self):
        return self.dice_v




def load_model(MODEL_SCALE = 16, DROP_R = 0.5, version='trained', model_type='evac_plus', base_path=None):

    class ChannelSum(layers.Layer):
        def __init__(self):
            super(ChannelSum, self).__init__()

        def call(self, inputs):
            return tf.reduce_sum(inputs, axis=-1, keepdims=True)

    def _diagonal_initializer(shape, *ignored, **ignored_too):
    return np.eye(shape[0], shape[1], dtype=np.float32)


    def _potts_model_initializer(shape, *ignored, **ignored_too):
        return -1 * _diagonal_initializer(shape)

    class CRF_RNN_Layer(layers.Layer):
        """ Implements the CRF-RNN layer.
        See https://github.com/sadeepj/crfasrnn_keras/blob/master/src/crfrnn_layer.py
        Based on GPU implementation here: https://github.com/MiguelMonteiro/CRFasRNNLayer
        Unaries and reference image must be provided in order: [unaries, ref_image]
        """

        def __init__(self,
                    image_dims,
                    num_classes,
                    theta_alpha,
                    theta_beta,
                    theta_gamma,
                    num_iterations,
                    **kwargs):
            self.image_dims = image_dims
            self.num_classes = num_classes
            self.theta_alpha = theta_alpha
            self.theta_beta = theta_beta
            self.theta_gamma = theta_gamma
            self.num_iterations = num_iterations
            self.spatial_ker_weights = None
            self.bilateral_ker_weights = None
            self.compatibility_matrix = None
            super(CRF_RNN_Layer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
                                                    shape=(self.num_classes,),
                                                    initializer=tf.initializers.truncated_normal(mean=1, stddev=1.0),
                                                    trainable=True)

            self.spatial_ker_weights = tf.linalg.diag(self.spatial_ker_weights)

            self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
                                                        shape=(self.num_classes,),
                                                        initializer=tf.initializers.truncated_normal(mean=1, stddev=1.0),
                                                        trainable=True)
            self.bilateral_ker_weights = tf.linalg.diag(self.bilateral_ker_weights)

            self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                        shape=(self.num_classes, self.num_classes),
                                                        initializer=tf.initializers.truncated_normal(mean=1, stddev=0.1),
                                                        trainable=True)

            super(CRF_RNN_Layer, self).build(input_shape)

        def call(self, inputs, **kwargs):
            unaries = inputs[0]
            reference_image = inputs[1]

            # Prepare filter normalization coefficients
            unaries_shape = tf.shape(unaries)
            q_values = unaries
            for i in range(self.num_iterations):
                q_values = tf.nn.softmax(q_values)

                # Spatial filtering
                spatial_out = custom_module.lattice_filter(q_values, reference_image, bilateral=False,
                                                    theta_gamma=self.theta_gamma)

                # Bilateral filtering
                bilateral_out = custom_module.lattice_filter(q_values, reference_image, bilateral=True,
                                                    theta_alpha=self.theta_alpha, theta_beta=self.theta_beta)

                # Weighting filter outputs
                message_passing = tf.matmul(self.spatial_ker_weights,
                                            tf.transpose(tf.reshape(spatial_out, (-1, self.num_classes)))) + \
                                tf.matmul(self.bilateral_ker_weights,
                                            tf.transpose(tf.reshape(bilateral_out, (-1, self.num_classes))))

                # Compatibility transform
                pairwise = tf.matmul(self.compatibility_matrix, message_passing)

                # Adding unary potentials
                pairwise = tf.reshape(tf.transpose(pairwise), unaries_shape)
                q_values = unaries - pairwise

            return q_values

        def compute_output_shape(self, input_shape):
            return input_shape

    if version != 'trained':
        # input and encoder block1      
        inputs = tf.keras.Input(shape=(128, 128, 128, 1))
        raw_input_2 = tf.keras.Input(shape=(64, 64, 64, 1))
        raw_input_3 = tf.keras.Input(shape=(32, 32, 32, 1))
        raw_input_4 = tf.keras.Input(shape=(16, 16, 16, 1))
        raw_input_5 = tf.keras.Input(shape=(8, 8, 8, 1))
        x = layers.Conv3D(MODEL_SCALE, (5, 5, 5), padding='same', kernel_initializer=initializer)(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv3D(MODEL_SCALE, (5, 5, 5), padding='same', kernel_initializer=initializer)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LayerNormalization()(x)
        channel_input = layers.ReLU()(x)

        # residual connection step
        fwd_1 = ChannelSum()(channel_input)
        fwd_1 = layers.Add()([fwd_1, inputs])
        # lowering resolution to encoder block2
        input_2 = layers.Conv3D(1, (2, 2, 2), strides=2, padding='same', kernel_initializer=initializer)(fwd_1)
        input_2 = layers.ReLU()(input_2)
        input_2 = layers.Concatenate()([input_2, raw_input_2])
        x = layers.Conv3D(MODEL_SCALE*2, (5, 5, 5), padding='same', kernel_initializer=initializer)(input_2)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(MODEL_SCALE*2, (5, 5, 5), padding='same', kernel_initializer=initializer)(x)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)

        fwd_2 = ChannelSum()(x)
        fwd_2 = layers.Add()([fwd_2, input_2])

        # lowering resolution to encoder block3
        input_3 = layers.Conv3D(1, (2, 2, 2), strides=2, padding='same', kernel_initializer=initializer)(fwd_2)
        input_3 = layers.ReLU()(input_3)
        input_3 = layers.Concatenate()([input_3, raw_input_3])
        x = layers.Conv3D(MODEL_SCALE*4, (5, 5, 5), padding='same', kernel_initializer=initializer)(input_3)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(MODEL_SCALE*4, (5, 5, 5), padding='same', kernel_initializer=initializer)(x)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(MODEL_SCALE*4, (5, 5, 5), padding='same', kernel_initializer=initializer)(x)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)

        fwd_3 = ChannelSum()(x)
        fwd_3 = layers.Add()([fwd_3, input_3])

        # lowering resolution to encoder block4
        input_4 = layers.Conv3D(1, (2, 2, 2), strides=2, padding='same', kernel_initializer=initializer)(fwd_3)
        input_4 = layers.ReLU()(input_4)
        input_4 = layers.Concatenate()([input_4, raw_input_4])
        x = layers.Conv3D(MODEL_SCALE*8, (5, 5, 5), padding='same', kernel_initializer=initializer)(input_4)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(MODEL_SCALE*8, (5, 5, 5), padding='same', kernel_initializer=initializer)(x)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(MODEL_SCALE*8, (5, 5, 5), padding='same', kernel_initializer=initializer)(x)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)

        fwd_4 = ChannelSum()(x)
        fwd_4 = layers.Add()([fwd_4, input_4])

        # lowering resolution to latent block
        input_5 = layers.Conv3D(1, (2, 2, 2), strides=2, padding='same', kernel_initializer=initializer)(fwd_4)
        input_5 = layers.ReLU()(input_5)
        input_5 = layers.Concatenate()([input_5, raw_input_5])
        x = layers.Conv3D(MODEL_SCALE*16, (5, 5, 5), padding='same', kernel_initializer=initializer)(input_5)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(MODEL_SCALE*16, (5, 5, 5), padding='same', kernel_initializer=initializer)(x)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(MODEL_SCALE*16, (5, 5, 5), padding='same', kernel_initializer=initializer)(x)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)

        # increasing resolution
        up_4 = ChannelSum()(x)
        up_4 = layers.Add()([up_4, input_5])
        up_4 = layers.Conv3DTranspose(1, (2, 2, 2), strides=2, padding='same', kernel_initializer=initializer)(up_4)
        up_4 = layers.ReLU()(up_4)

        # decoder block 4 with information from encoder
        recv_4 = layers.Concatenate()([fwd_4, up_4])
        x = layers.Conv3D(MODEL_SCALE*8, (5, 5, 5), padding='same', kernel_initializer=initializer)(recv_4)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(MODEL_SCALE*8, (5, 5, 5), padding='same', kernel_initializer=initializer)(x)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(MODEL_SCALE*8, (5, 5, 5), padding='same', kernel_initializer=initializer)(x)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)

        up_3 = ChannelSum()(x)
        up_3 = layers.Add()([up_3, up_4])
        up_3 = layers.Conv3DTranspose(1, (2, 2, 2), strides=2, padding='same', kernel_initializer=initializer)(up_3)
        up_3 = layers.ReLU()(up_3)

        # decoder block 3 with information from encoder
        recv_3 = layers.Concatenate()([fwd_3, up_3])
        x = layers.Conv3D(MODEL_SCALE*4, (5, 5, 5), padding='same', kernel_initializer=initializer)(recv_3)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(MODEL_SCALE*4, (5, 5, 5), padding='same', kernel_initializer=initializer)(x)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(MODEL_SCALE*4, (5, 5, 5), padding='same', kernel_initializer=initializer)(x)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)

        up_2 = ChannelSum()(x)
        up_2 = layers.Add()([up_2, up_3])
        up_2 = layers.Conv3DTranspose(1, (2, 2, 2), strides=2, padding='same', kernel_initializer=initializer)(up_2)
        up_2 = layers.ReLU()(up_2)

        # decoder block 2 with information from encoder
        recv_2 = layers.Concatenate()([fwd_2, up_2])
        x = layers.Conv3D(MODEL_SCALE*2, (5, 5, 5), padding='same', kernel_initializer=initializer)(recv_2)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(MODEL_SCALE*2, (5, 5, 5), padding='same', kernel_initializer=initializer)(x)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)

        up_1 = ChannelSum()(x)
        up_1 = layers.Add()([up_1, up_2])
        up_1 = layers.Conv3DTranspose(1, (2, 2, 2), strides=2, padding='same', kernel_initializer=initializer)(up_1)
        up_1 = layers.ReLU()(up_1)

        # decoder block 1 with information from encoder
        recv_1 = layers.Concatenate()([fwd_1, up_1])
        x = layers.Conv3D(MODEL_SCALE, (5, 5, 5), padding='same', kernel_initializer=initializer)(recv_1)
        x = layers.Dropout(DROP_R)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)

        # last step to create the segmentation map
        pred = ChannelSum()(x)
        pred = layers.Add()([pred, up_1])
        pred = layers.Conv3D(2, (1, 1, 1), padding='same', kernel_initializer=initializer)(pred)
        crf_input = layers.Concatenate()([inputs, channel_input[..., :MODEL_SCALE//2]])
        output = CRF_RNN_Layer(image_dims=(128, 128, 128),
                            num_classes=2,
                            theta_alpha=1.,
                            theta_beta=2.,
                            theta_gamma=1.,
                            num_iterations=20,
                            name='crfrnn')([pred, crf_input])
        output = layers.Softmax()(output)

        model = Model({"input_1":inputs, "input_2":raw_input_2, "input_3":raw_input_3, "input_4":raw_input_4, "input_5":raw_input_5}, output[..., 0])
        return model

    if model_type == 'evac':
        model = tf.keras.models.load_model('trained_model/evac/', custom_objects={'dice_coef': dice_coef})
    elif model_type == 'evac_plus':
        model = tf.keras.models.load_model('trained_model/evac_plus/', custom_objects={'comb_loss': comb_loss, 'dice_coef_orig': dice_coef_orig, 'dice_coef_real': dice_coef_real, 'dice_coef_comp': dice_coef_comp})
    elif model_type == 'user':
        try:
            model = tf.keras.models.load_model(base_path, custom_objects={'dice_coef': dice_coef})
        except:
            model = tf.keras.models.load_model(base_path, custom_objects={'comb_loss': comb_loss})
    return model

def test_model(model, input_dir, output_dir, batch_size=1):
    if os.path.isdir(input_dir) == False:
        image_data, new_affine, shape = __prepare_img(input_dir)
        inputs = np.expand_dims(image_data, 0)
        inputs = {"input_1": inputs,
                  "input_2": resize(inputs, (inputs.shape[0], 64, 64, 64, 1)), 
                  "input_3": resize(inputs, (inputs.shape[0], 32, 32, 32, 1)), 
                  "input_4": resize(inputs, (inputs.shape[0], 16, 16, 16, 1)),
                  "input_5": resize(image_data, (inputs.shape[0], 8, 8, 8, 1))}

        prediction = model.predict(inputs)
        pred_output = np.reshape(prediction, (128,128,128))       

        pred_output, old_affine = __recover(pred_output, new_affine, shape)
        i = np.where(pred_output >= 0.5)
        j = np.where(pred_output < 0.5)
        pred_output[i] = 1.0
        pred_output[j] = 0.0
        pred_output = __largest(np.abs(1-__largest(np.abs(1-pred_output))))
        save_nifti(output_dir, pred_output, old_affine)
        print(input_dir + " processed")
        return

    inputs = np.zeros((batch_size, 128, 128, 128, 1))
    new_affines = np.zeros((batch_size, 4, 4))
    file_names = []
    idx = 0
    n_files = len(os.listdir(input_dir))
    for i, file_path in enumerate(sorted(os.listdir(input_dir))):
        image_data, new_affine, shape = __prepare_img(os.path.join(input_dir, file_path))
        shape = image_data.shape
        inputs[idx] = np.reshape(image_data, (128, 128, 128, 1))
        new_affines[idx] = new_affine
        file_names.append(file_path)
        if batch_size != 1 and idx != batch_size:
            if i != n_files-1:
                idx += 1
                continue
            else:
                inputs = inputs[:idx+1]
                new_affines = new_affines[:idx+1]
          
        inputs = {"input_1": inputs,
                  "input_2": resize(inputs, (inputs.shape[0], 64, 64, 64, 1)), 
                  "input_3": resize(inputs, (inputs.shape[0], 32, 32, 32, 1)), 
                  "input_4": resize(inputs, (inputs.shape[0], 16, 16, 16, 1)),
                  "input_5": resize(image_data, (inputs.shape[0], 8, 8, 8, 1))}

        prediction = model.predict(inputs)
        for b_idx in range(prediction.shape[0]):
            pred_output = np.reshape(prediction[b_idx], (128,128,128))       
            pred_output, old_affine = __recover(pred_output, new_affines[b_idx], shape)
            i = np.where(pred_output >= 0.5)
            j = np.where(pred_output < 0.5)
            pred_output[i] = 1.0
            pred_output[j] = 0.0
            pred_output = __largest(np.abs(1-__largest(np.abs(1-pred_output))))
            save_nifti(os.path.join(output_dir, file_names[b_idx]), pred_output, old_affine)
            print(file_path + " processed")
            
        inputs = np.zeros((batch_size, 128, 128, 128, 1))
        new_affines = np.zeros((batch_size, 4, 4))
        file_names = []
        idx = 0

def train_model(model, train_ds, val_ds = None, pre_trained=False, model_type='evac_plus', model_path, batch_size=1, l_r=0.01, l2_w=0.001):
    class dice_coef(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
            y_pred = tf.cast(y_pred, tf.float32)
            y_true = tf.cast(y_true, y_pred.dtype)
            pred_ssum = tf.math.reduce_sum(tf.math.square(y_pred), (1, 2, 3))
            target_ssum = tf.math.reduce_sum(tf.math.square(y_true), (1, 2, 3))
            mul_sum = tf.math.reduce_sum(tf.math.multiply_no_nan(y_pred, y_true), (1, 2, 3))
            
            res = tf.math.divide_no_nan(-2 * mul_sum, tf.math.add(pred_ssum, target_ssum))
            return 1 + res

    class comb_loss(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
            y_pred = tf.cast(y_pred, tf.float32)
            before_mask = y_pred[..., 2]
            after_mask = y_pred[..., 0]
            y_true = tf.cast(y_true, y_pred.dtype)
            pred_ssum = tf.math.reduce_sum(tf.math.square(after_mask), (1, 2, 3))
            target_ssum = tf.math.reduce_sum(tf.math.square(y_true), (1, 2, 3))
            mul_sum = tf.math.reduce_sum(tf.math.multiply_no_nan(after_mask, y_true), (1, 2, 3))
            res = tf.clip_by_value(1 + tf.math.divide_no_nan(-2 * mul_sum, tf.math.add(pred_ssum, target_ssum)), 0, 1)

            shape = tf.shape(y_true)
            flat_before = tf.reshape(before_mask, (shape[0], shape[1] * shape[2] * shape[3]))
            flat_after = tf.reshape(after_mask, (shape[0], shape[1] * shape[2] * shape[3]))
            pred_ssum = tf.math.reduce_sum(tf.math.square(before_mask), (1, 2, 3))
            target_ssum = tf.math.reduce_sum(tf.math.square(after_mask), (1, 2, 3))
            mul_sum = tf.math.reduce_sum(tf.math.multiply_no_nan(before_mask, after_mask), (1, 2, 3))
            res2 = tf.clip_by_value(1 + tf.math.divide_no_nan(-2 * mul_sum, tf.math.add(pred_ssum, target_ssum)), 0, 1)

            return res - res2 * l2_w

    if pre_trained == False:
        optimizer = tf.keras.optimizers.Adam(l_r)
        if model_type == 'evac':
            model.compile(loss=dice_coef(reduction=tf.keras.losses.Reduction.AUTO), optimizer=optimizer)
        else:
            model.compile(loss=comb_loss(reduction=tf.keras.losses.Reduction.AUTO), optimizer=optimizer)

    checkpoint_path = 'tmp/' + model_type
    if os.path.exists('tmp') == False:
        os.makedirs('tmp')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                        save_weights_only=True,
                                        save_best_only=True)
    ]

    if val_ds != None:
        model.fit(x=train_ds, epochs=EPOCHS, callbacks=callbacks)
    else:
        model.fit(x=train_ds, validation_data = val_ds, epochs=EPOCHS, callbacks=callbacks)
    model.load_weights(checkpoint_path)
    model.save(model_path, save_format='tf')
    

from joblib.numpy_pickle_utils import xrange
from tensorflow import keras
import keras_contrib
import keras_contrib.layers


def model_init(opts):
    """Simple sequential image-to-image convolutional neural network"""
    
    init_fn = keras.initializers.VarianceScaling(2.)

    model = keras.models.Sequential()
    isFirst = True
    for ks, nk, a in zip(opts.kernelSizes, opts.numKernels, opts.activations):

        if isFirst:
            model.add(keras.layers.Conv2D(nk, kernel_size=ks,strides=opts.strides,padding=opts.padding,
                                kernel_initializer=init_fn, input_shape=opts.inputShape))
            isFirst = False
        else: 
            model.add(keras.layers.Conv2D(nk, kernel_size=ks,strides=opts.strides,padding=opts.padding,
                                kernel_initializer=init_fn))

        if opts.includeInsNormLayer: 
            model.add(keras_contrib.layers.InstanceNormalization(axis=opts.insNormAxis))

        model.add(keras.layers.Activation(a))
        if opts.dropRate > 0.0: 
            model.add(keras.layers.Dropout(rate=opts.dropRate))
    return model


def model_unet(opts): 
    init_fn = keras.initializers.VarianceScaling(2.)
    model_input = keras.layers.Input(shape=opts.inputShape)
    prev = model_input
    scaleSpace = []

    # downsampling path
    for i in xrange(opts.numScales): 
        out = keras.layers.Conv2D(filters=opts.numKernels, kernel_size=opts.kernelSizes, strides=opts.strides, padding='same',
                    kernel_initializer=init_fn,activation=opts.activations)(prev)
        scaleSpace.append(out)
        
        prev = keras.layers.MaxPool2D(pool_size=opts.poolSize, strides=opts.poolStrides, padding=opts.poolPadding)(out)

    # base case
    prev = keras.layers.Conv2D(filters=opts.numKernels, kernel_size=opts.kernelSizes, strides=opts.strides, padding='same',
                        kernel_initializer=init_fn,activation=opts.activations)(prev)
    
    # upsampling path
    for i in xrange(opts.numScales): 
        out = keras.layers.Conv2DTranspose(filters=opts.numKernels,kernel_size=opts.kernelSizes, strides=2, padding='same',
                                    kernel_initializer=init_fn,)(prev)
 
        out = keras.layers.concatenate(inputs=[out, scaleSpace[-1-i]], axis=-1)
        prev = keras.layers.Conv2D(filters=opts.numKernels, kernel_size=opts.kernelSizes, strides=opts.strides, padding='same',
                    kernel_initializer=init_fn,activation=opts.activations)(out)
    
    model_output = keras.layers.Conv2D(filters=1, kernel_size=opts.kernelSizes, strides=opts.strides, padding='same',
                    kernel_initializer=init_fn,activation='sigmoid')(prev)

    model = keras.models.Model(inputs=(model_input,), outputs=(model_output,))
    return model


models_dict = dict()
models_dict['imageToImageSeq'] = model_init
models_dict['imageToImageUnet'] = model_unet
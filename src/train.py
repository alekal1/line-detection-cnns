import argparse

import keras_contrib
import numpy as np
import os 
from time import gmtime, strftime

from joblib.numpy_pickle_utils import xrange
from tensorflow import keras
import pickle
import skimage.io as io
import os
import numpy as np
from skimage import img_as_float
import keras.backend as K

def train(opts): 
	"""Performs the whole algorithm i.e trains a given neural network on given data using given learning parameters
	
	Args: 
	opts: command line arguments
	
	Returns: 
	None"""

	#Creating given model OR loading pretrained network..
	if opts.loadModel is None: 
		print('Creating Network Architecture..')
		model = models_dict[opts.netType](opts)
	else: 
		print('Loading Pretrained Network from {}..'.format(opts.loadModel))
		model = keras.models.load_model(opts.loadModel,compile=False)

	#Compiling given model using given learning parameters..
	optimizer = optimizers_dict[opts.optimizerType](lr=opts.learningRate, decay=opts.lrDecay)
	model.compile(optimizer=optimizer, loss=losses_dict[opts.lossType](model.input, opts.epsilon))

	#Configuring data loaders/generators now..
	train_generator = generators_dict[opts.generatorType](os.path.join(opts.dataDir,'train', opts.dataType),
														 opts.ext, opts.batchSize, None, mode='train')
	val_generator = generators_dict[opts.generatorType](os.path.join(opts.dataDir,'val', opts.dataType), 
														opts.ext, opts.batchSize*2, None, mode='validation')
														
	steps_per_epoch = (len(os.listdir(os.path.join(opts.dataDir,'train', opts.dataType, 'X'))) / opts.batchSize) / opts.logPerEpoch
	validation_steps = len(os.listdir(os.path.join(opts.dataDir,'val', opts.dataType, 'X'))) / (opts.batchSize*2)
	validation_steps = 1 if (validation_steps == 0) else validation_steps
	numEpochs_ = opts.numEpochs * opts.logPerEpoch

	#Configuring experimentation directories..
	if not os.path.exists(opts.expDir): 
		os.makedirs(opts.expDir)
		writeConfigToFile(os.path.join(opts.expDir,'opts.txt'), vars(opts), model)
		# keras.utils.plot_model(model, to_file=os.path.join(opts.expDir,'network.png'), show_layer_names=False)
	
	#Configuring callbacks..
	os.makedirs(os.path.join(opts.expDir, 'model'))
	ckptCallback=keras.callbacks.ModelCheckpoint(os.path.join(opts.expDir,'model', '{epoch:02d}-{loss:.2f}.hdf5'),
								monitor='loss',save_best_only=True)
	tboardCallback=keras.callbacks.TensorBoard(log_dir=os.path.join(opts.expDir,'tensorboardLogs'))
	valsaver = valImagesSaver(dataDir=os.path.join(opts.dataDir,'val', opts.dataType, 'X'),
					ext=opts.ext, outDir=os.path.join(opts.expDir, 'valImages'))
	terminator = keras.callbacks.TerminateOnNaN()

	#FINALLY! TRAINING NOW..
	history = model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=numEpochs_,
								verbose=opts.verbosity, validation_data=val_generator, validation_steps=validation_steps,
								callbacks=[ckptCallback,tboardCallback,valsaver,terminator])

	with open(os.path.join(opts.expDir, 'trainHistory.pkl'), 'wb') as fobj: 
		pickle.dump(history.history, fobj)
	return


def SetArguments(parser): 
	#Data loading arguments
	parser.add_argument('-dataDir',action='store', type=str, default='../data/', dest='dataDir')
	parser.add_argument('-dataType',action='store', type=str, default='noNoise', dest='dataType')
	parser.add_argument('-ext', action='store',type=list, default=['png', 'jpg'], dest='ext')
	parser.add_argument('-generatorType', action='store', type=str, default='generator_full_image', dest='generatorType')
	parser.add_argument('-inputShape', action='store', type=tuple, default=(512,512,1), dest='inputShape')

	#Model parameters
	parser.add_argument('-netType', action='store', type=str, default='imageToImageSeq', dest='netType')
	parser.add_argument('-loadModel', action='store', type=str, default=None, dest='loadModel')
	
	parser.add_argument('-dropRate', action='store', type=float, default=0.0, dest='dropRate')
	parser.add_argument('-kernelSizes', action='store', type=str, default='3,3', dest='kernelSizes')
	parser.add_argument('-numKernels', action='store', type=str, default='16,16', dest='numKernels')
	parser.add_argument('-activations', action='store', type=str, default='relu,sigmoid', dest='activations')
	parser.add_argument('-padding', action='store', type=str, default='same', dest='padding')
	parser.add_argument('-strides', action='store', type=int, default=1, dest='strides')
	parser.add_argument('-includeInsNormLayer', action='store', type=bool, default=False, dest='includeInsNormLayer')
	parser.add_argument('-insNormAxis', action='store', type=int, default=None, dest='insNormAxis')

	parser.add_argument('-numScales', action='store', type=int, default=3, dest='numScales')
	parser.add_argument('-poolSize', action='store', type=int, default=2, dest='poolSize')
	parser.add_argument('-poolStrides', action='store', type=int, default=2, dest='poolStrides')
	parser.add_argument('-poolPadding', action='store', type=str, default='valid', dest='poolPadding')

	#Learning parameters
	parser.add_argument('-optimizerType', action='store', type=str, default='adam', dest='optimizerType')
	parser.add_argument('-learningRate', action='store', type=float, default=1e-3, dest='learningRate')
	parser.add_argument('-lrDecay', action='store', type=float, default=0.0, dest='lrDecay')
	parser.add_argument('-numEpochs', action='store', type=int, default=1, dest='numEpochs')
	parser.add_argument('-verbosity', action='store', type=int, default=1, dest='verbosity')
	parser.add_argument('-batchSize', action='store', type=int, default=16, dest='batchSize')

	#Loss function parameters
	parser.add_argument('-lossType', action='store', type=str, default='weightedBinaryCrossEntropy', dest='lossType')
	parser.add_argument('-epsilon', action='store', type=float, default=1e-6, dest='epsilon')

	#Logging parameters
	parser.add_argument('-logRootDir',action='store',type=str, default='../experiments/',dest='logRootDir')
	parser.add_argument('-logDir',action='store',type=str, default=strftime("%d-%m-%Y__%H-%M-%S",gmtime()),dest='logDir')
	parser.add_argument('-logPerEpoch',action='store',type=int, default=1,dest='logPerEpoch')
	return

def PostprocessOpts(opts): 
	opts.kernelSizes = [int(x) for x in opts.kernelSizes.split(',')]
	opts.numKernels = [int(x) for x in opts.numKernels.split(',')]
	opts.activations = opts.activations.split(',')

	opts.kernelSizes = opts.kernelSizes[0] if (len(opts.kernelSizes)==1) else opts.kernelSizes
	opts.numKernels = opts.numKernels[0] if (len(opts.numKernels)==1) else opts.numKernels
	opts.activations = opts.activations[0] if (len(opts.activations)==1) else opts.activations

	opts.expDir = os.path.join(opts.logRootDir, opts.logDir)
	return


def model_init(opts):
	"""Simple sequential image-to-image convolutional neural network"""

	init_fn = keras.initializers.VarianceScaling(2.)

	model = keras.models.Sequential()
	isFirst = True
	for ks, nk, a in zip(opts.kernelSizes, opts.numKernels, opts.activations):

		if isFirst:
			model.add(keras.layers.Conv2D(nk, kernel_size=ks, strides=opts.strides, padding=opts.padding,
										  kernel_initializer=init_fn, input_shape=opts.inputShape))
			isFirst = False
		else:
			model.add(keras.layers.Conv2D(nk, kernel_size=ks, strides=opts.strides, padding=opts.padding,
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
		out = keras.layers.Conv2D(filters=opts.numKernels, kernel_size=opts.kernelSizes, strides=opts.strides,
								  padding='same',
								  kernel_initializer=init_fn, activation=opts.activations)(prev)
		scaleSpace.append(out)

		prev = keras.layers.MaxPool2D(pool_size=opts.poolSize, strides=opts.poolStrides, padding=opts.poolPadding)(out)

	# base case
	prev = keras.layers.Conv2D(filters=opts.numKernels, kernel_size=opts.kernelSizes, strides=opts.strides,
							   padding='same',
							   kernel_initializer=init_fn, activation=opts.activations)(prev)

	# upsampling path
	for i in xrange(opts.numScales):
		out = keras.layers.Conv2DTranspose(filters=opts.numKernels, kernel_size=opts.kernelSizes, strides=2,
										   padding='same',
										   kernel_initializer=init_fn, )(prev)

		out = keras.layers.concatenate(inputs=[out, scaleSpace[-1 - i]], axis=-1)
		prev = keras.layers.Conv2D(filters=opts.numKernels, kernel_size=opts.kernelSizes, strides=opts.strides,
								   padding='same',
								   kernel_initializer=init_fn, activation=opts.activations)(out)

	model_output = keras.layers.Conv2D(filters=1, kernel_size=opts.kernelSizes, strides=opts.strides, padding='same',
									   kernel_initializer=init_fn, activation='sigmoid')(prev)

	model = keras.models.Model(inputs=(model_input,), outputs=(model_output,))
	return model


models_dict = dict()
models_dict['imageToImageSeq'] = model_init
models_dict['imageToImageUnet'] = model_unet

optimizers_dict = dict()
optimizers_dict['adam'] = keras.optimizers.Adam


def generator_full_image(directory, ext, batch_size, preprocessing=None, mode='train'):
	""""""
	# loading filenames (only and not the image)
	fnamesX = sorted(os.listdir(os.path.join(directory, 'X')))
	pathnamesX = [os.path.join(directory, 'X', f) for f in fnamesX if f.split('.')[-1] in ext]
	pathnamesX = np.array(pathnamesX)

	fnamesY = sorted(os.listdir(os.path.join(directory, 'Y')))
	pathnamesY = [os.path.join(directory, 'Y', f) for f in fnamesY if f.split('.')[-1] in ext]
	pathnamesY = np.array(pathnamesY)

	idx = np.arange(pathnamesX.shape[0])
	while True:

		# shuffling at the start of each epoch..
		np.random.shuffle(idx)

		for i in xrange(0, pathnamesX.shape[0], batch_size):

			# picking a batch..
			pathnamesX_batch = pathnamesX[idx[i:i + batch_size]]
			# pathnamesY_batch = pathnamesY[idx[i:i + batch_size]]

			imagesX = io.ImageCollection(pathnamesX_batch)
			# imagesY = io.ImageCollection(pathnamesY_batch)

			# optionally applying preporcessing to each batch
			# if preprocessing is not None:
				# imagesX, imagesY = preprocessing(imagesX, imagesY)

			imagesX = io.concatenate_images(imagesX)
			# imagesY = io.concatenate_images(imagesY)

			imagesX = img_as_float(imagesX[:, :, :, np.newaxis])
			# imagesY = img_as_float(imagesY[:, :, :, np.newaxis]).astype(bool)

			yield imagesX # imagesY


generators_dict = dict()
generators_dict['generator_full_image'] = generator_full_image


def BinaryCrossEntropy(y_true, y_pred):
	losses = -((y_true * K.log(y_pred)) + ((1 - y_true) * K.log(1 - y_pred)))
	return K.mean(losses)


def WeightedBinaryCrossEntropy(x_true, eps):
	def WeightedBinaryCrossEntropy_(y_true, y_pred):
		err = -((y_true * K.log(y_pred)) + ((1 - y_true) * K.log(1 - y_pred)))

		probs = K.mean(x_true, axis=(1, 2, 3), keepdims=True)
		weights_pos, weights_neg = 1. / (probs + eps), 1. / ((1 - probs) + eps)
		weights = (x_true * weights_pos) + ((1 - x_true) * weights_neg)

		return K.mean(err * weights)

	return WeightedBinaryCrossEntropy_


def L2Loss(y_true, y_pred):
	return K.mean(K.square(y_true - y_pred))


def WeightedL2Loss(x_true, eps):
	def WeightedL2Loss(y_true, y_pred):
		err = K.square(y_true - y_pred)
		probs = K.mean(x_true, axis=(1, 2, 3), keepdims=True)
		weights_pos, weights_neg = 1. / (probs + eps), 1. / ((1 - probs) + eps)
		weights = (x_true * weights_pos) + ((1 - x_true) * weights_neg)

		return K.mean(weights * err)

	return WeightedL2Loss


losses_dict = {'binaryCrossEntropy': BinaryCrossEntropy, 'weightedBinaryCrossEntropy': WeightedBinaryCrossEntropy,
			   'l2Loss': L2Loss, 'weightedL2Loss': WeightedL2Loss}

def CheckAndCreate(path):
    if not os.path.exists(path):
        os.makedirs(path)


class valImagesSaver(keras.callbacks.Callback):
    def __init__(self, dataDir, ext, outDir):
        keras.callbacks.Callback.__init__(self)
        self.min = np.inf
        self.dataDir = dataDir
        self.outDir = outDir

        CheckAndCreate(outDir)

        fnamesX = sorted(os.listdir(self.dataDir))
        pathnamesX = [os.path.join(dataDir,f) for f in fnamesX if f.split('.')[-1] in ext]
        self.pathnamesX = np.array(pathnamesX)

    def on_epoch_end(self, epoch, logs):

        # If best validation model yet..
        if logs.get('val_loss') < self.min:
            self.min = logs.get('val_loss')

            imgs = io.ImageCollection(self.pathnamesX)
            imgs = img_as_float(imgs.concatenate()[:,:,:,np.newaxis])

            ypreds = self.model.predict(imgs,batch_size=1)

            for i, path in enumerate(self.pathnamesX):
                fname = path.split('/')[-1]
                outPath = os.path.join(self.outDir, fname)

                io.imsave(outPath, ypreds[i,:,:,0])


def writeConfigToFile(fpath, optsDict, model):
    fobj = open(fpath, 'w')

    for k, v in optsDict.items():
        fobj.write('{} >> {}\n'.format(str(k), str(v)))
    fobj.close()

if __name__=='__main__': 
	parser = argparse.ArgumentParser()
	SetArguments(parser)

	opts = parser.parse_args()
	PostprocessOpts(opts)

	train(opts)

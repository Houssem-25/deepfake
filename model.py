from keras import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import * 
from keras.optimizers import *
from keras.losses import categorical_crossentropy, binary_crossentropy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Model as MD
import os 

log_dir = os.getcwd() +"/models_log"
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
	monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

class Model():
	def __init__(self, image_train,audio_train,image_validation,audio_validation,epochs=10):
		self.image_train =image_train
		self.audio_train = audio_train
		self.image_validation = image_validation
		self.audio_validation =audio_validation
		self.epochs= epochs
		#self.batch_size = image_train.batch_size
	
	def create_model(self):
		image_model_,image_input_ = self.image_model()
		audio_model_,audio_input_ = self.audio_model()
		x = concatenate([image_model_,audio_model_])
		x = Dense(128,activation="relu")(x)
		x = Dense(32,activation="relu")(x)
		x = Dense(16,activation="relu")(x)
		main_output = Dense(2, activation='softmax', name='main_output')(x)
		self.model = MD(inputs=[image_input_, audio_input_], outputs=main_output)
		opt = SGD(0.01)
		self.model.compile(optimizer=opt,loss=categorical_crossentropy,metrics=['accuracy'])
		
	def audio_model(self):
		audio_shape = (20,431)
		audio_input_ = Input(shape=(audio_shape))
		audio_model_ = Conv1D(256,kernel_size=7)(audio_input_)
		audio_model_ = Activation('relu')(audio_model_)
		audio_model_ = BatchNormalization(axis=-1)(audio_model_)
		audio_model_ = Flatten()(audio_model_)
		audio_model_  = Dense(512)(audio_model_)
		return audio_model_,audio_input_

	def image_model(self):
		image_shape = (300,200,200,3)
		image_input_ = Input(shape=(image_shape))
		image_model_ = Conv3D(8, (3,3,3),strides=(2,2,2), padding='same')(image_input_)
		image_model_ =Activation('relu')(image_model_)
		image_model_ =BatchNormalization(axis=-1)(image_model_)
		image_model_ =Conv3D(16, (5,5,5),strides=(2,2,2), padding='valid')(image_model_)
		image_model_ =Activation('relu')(image_model_)
		image_model_ =BatchNormalization(axis=-1)(image_model_)
		image_model_ =Reshape(target_shape=(1024, 876, 3))(image_model_)
		image_model_ =Conv2D(32, (3,3),strides=(2,2), padding='valid')(image_model_)
		image_model_ =Activation('relu')(image_model_)
		image_model_ =BatchNormalization(axis=-1)(image_model_)
		image_model_ =Conv2D(32, (3,3),strides=(2,2), padding='valid')(image_model_)
		image_model_ =Activation('relu')(image_model_)
		image_model_ =BatchNormalization(axis=-1)(image_model_)
		image_model_ =Conv2D(64, (3,3),strides=(2,2), padding='valid')(image_model_)
		image_model_ =Activation('relu')(image_model_)
		image_model_ =BatchNormalization(axis=-1)(image_model_)
		image_model_ =Flatten()(image_model_)
		return image_model_ , image_input_
		
	def summery(self):
		self.model.summary()

	def fit(self):

		self.model.fit_generator(generator=self.train_generator,epochs=self.epochs,
                    			validation_data=self.validation_generator,
                    			use_multiprocessing=True,
                    			workers=4,callbacks = [checkpoint,reduce_lr,early_stopping])
		self.model.save("save")

	def evaluate(self):
		score, acc = self.model.evaluate_generator(generator=self.test_generator,
                                              max_queue_size=10, workers=1, use_multiprocessing=False,
                                              verbose=0)
		print('Test score:', score)
		print('Test accuracy:', acc)



model = Model([],[],[],[])
model.create_model()
model.summery()
model.model.save("ss")
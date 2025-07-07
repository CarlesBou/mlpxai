# -*- coding: utf-8 -*-

"""
Jose L. Carles - Enrique Carmona - UNED - 2024-2025
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlpxai.explainers.face.kerasface import FACEExplainer

from mlpxai.utils.visualize import plot_MNIST_digit

import keras
from keras.datasets import mnist
from keras.models import Model


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf

import numpy as np
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



'''
CLASSIFICATION WITH IMAGES - MNIST
'''
seed = 1
# np.seed = 1
# random.seed = 1
# tf.random.set_seed(1)
keras.utils.set_random_seed(seed)

    
ds_name = 'MNIST'


(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_inputs = 28 * 28
num_outputs = 10

dataset_name = 'MNIST'

input_layer = Input(shape=(num_inputs,))
hidden_layer = Dense(num_inputs * 2, activation='relu')(input_layer)
hidden_layer = Dense(5, activation='relu')(hidden_layer)
output_layer = Dense(num_outputs, activation='linear')(hidden_layer)

model_version = 0

validation_split = 0.1 
epochs = 10

use_saved_model_weights = True

    
X_train = X_train.astype(np.float32) / 255.   #Transform integer pixel values to [0,1]
X_train = X_train.reshape(-1, num_inputs)     #Transfor image matrix into vector
X_test = X_test.astype(np.float32) / 255.     #Transform integer pixel values to [0,1]
X_test = X_test.reshape(-1, num_inputs)       #Transfor image matrix into vector
    
y_train_categorical = to_categorical(y_train, num_outputs).astype(np.float32)
    

''' 
Create and compile the model
'''
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='nadam',
              metrics=['accuracy'],
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))


print(f'Using dataset: {dataset_name}, {len(y_train) + len(y_test)} samples ({len(y_train)} train / {len(y_test)} test), {num_inputs} features, {num_outputs} classes')

'''
As the MNIST dataset is quite big, try to load a 
pretrained model. If it doesn't exist, train and save a new one
'''
weights_file_name = f'{dataset_name}_classfication_seed_{seed}_epochs_{epochs}.weights.h5'

if use_saved_model_weights:
    if os.path.isfile(weights_file_name):
        model.load_weights(weights_file_name)
        print(
            f'Using pretrained classification model weights from file {weights_file_name}')
    else:
        print('Training classification model ... ', end='')
        my_fit = model.fit(X_train, y_train_categorical,
                           epochs=epochs, validation_split=validation_split, verbose=0)
        model.save_weights(weights_file_name)
        print('OK')
else:
    print('Training classification model ... ', end='')
    
    my_fit = model.fit(X_train, y_train_categorical,
                       epochs=epochs, validation_split=validation_split, verbose=0)
    print('OK')


'''
Run predictions over the test dataset and show the network performance
'''
print('Generating predictions for test data ... ', end='')
predictions = model.predict(X_test, verbose=0)

y_mlp = np.argmax(predictions, axis=1)

print('OK')

accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
print(f'Test data accuracy = {accuracy:.5f}\n')


''''
MNIST samples 
'''
samples = [8535, 8330, 8841, 23, 57, 8003]


'''
Create the FACE Explainer
'''
face = FACEExplainer(model)


for sample in samples:

    print(f'Computing FACE attributions for test sample {sample} ground/net/FACE={y_test[sample]}/{y_mlp[sample]}/', end='')
    
    '''
    And perform the explanation to get the feature attributions
    '''
    FACE_contrib = face.explain(X_test[sample])
        
    y_FACE = np.argmax(np.sum(FACE_contrib, axis=1))
    
    print(f'{y_FACE} ... OK')
    
    y_truth = y_test[sample]
        
    contrib = np.array([FACE_contrib[y_truth][1:]])
    
    plot_MNIST_digit(X_test[sample], contrib, resize=1)
    



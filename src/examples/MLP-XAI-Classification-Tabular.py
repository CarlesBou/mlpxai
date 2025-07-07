# -*- coding: utf-8 -*-

"""
Jose L. Carles - Enrique Carmona - UNED - 2024-2025
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from mlpxai.explainers.face.kerasface import FACEExplainer

from mlpxai.utils.visualize import plot_bar_contrib

import keras
from keras.models import Model


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf

import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# tf.keras.mixed_precision.set_global_policy('float64')

'''
CLASSIFICATION
'''

# np.seed = 1
# random.seed = 1
# tf.random.set_seed(1)
keras.utils.set_random_seed(1)


ds_name = 'liver'
liver_drinks = 7


'''
Liver disorder dataset - Classification

  Attribute information:
   1. mcv	mean corpuscular volume
   2. alkphos	alkaline phosphotase
   3. sgpt	alamine aminotransferase
   4. sgot 	aspartate aminotransferase
   5. gammagt	gamma-glutamyl transpeptidase
   6. drinks	number of half-pint equivalents of alcoholic beverages
                drunk per day --> Target. Dichotimeze with
                Class 0 as drinks < 7, Class 1 with drinks >= 7
   7. selector  field used to split data into two sets --> Ignored

'''

liver_df = pd.read_csv('sample_data/liver.csv', delimiter=';')

X = liver_df.iloc[:, :-2]
y = pd.DataFrame(liver_df['drinks'])

# Dichotomize target, number of drinks per day, in two classes
y.loc[y['drinks'] < liver_drinks] = 0
y.loc[y['drinks'] >= liver_drinks] = 1

y = y.values.ravel().astype(int)

feature_names = X.columns

num_inputs = X.shape[1]
num_outputs = 2

dataset_name = 'Liver disorder'

input_layer = Input(shape=(num_inputs,))
hidden_layer = Dense(30, activation='relu')(input_layer)
hidden_layer = Dense(5, activation='relu')(hidden_layer)    
output_layer = Dense(num_outputs, activation='linear')(hidden_layer)

# input_layer = Input(shape=(num_inputs,), dtype='float64')
# hidden_layer = Dense(30, activation='relu', dtype='float64')(input_layer)
# hidden_layer = Dense(5, activation='relu', dtype='float64')(hidden_layer)    
# output_layer = Dense(num_outputs, activation='linear', dtype='float64')(hidden_layer)

test_size = 0.15
validation_split = 0.10
epochs = 70

use_saved_model_weights = False



'''
Rescale the input data to [0, 1] with a MinMaxScaler. Later we will
use the scaler to transform back the data to the original domain
'''
scaler = MinMaxScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_size,
                                                    shuffle=True,
                                                    stratify=y)

y_train_categorical = to_categorical(y_train, num_outputs).astype(np.float32)


X_test = X_test.to_numpy()

''' 
Create and compile the model
'''
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='nadam',
              metrics=['accuracy'],
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))


print(f'Using dataset: {dataset_name}, {len(y)} samples ({len(y_train)} train / {len(y_test)} test), {num_inputs} features, {num_outputs} classes')


'''
Train the model
'''    
print('Training classification model ... ', end='')

my_fit = model.fit(X_train, y_train_categorical,
                    epochs=epochs, validation_split=validation_split, verbose=0)

print('OK')


'''
Generating predictions
'''
print('Generating predictions for test data ... ', end='')

predictions = model.predict(X_test, verbose=0)

y_mlp = np.argmax(predictions, axis=1)

print('OK')

accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)

print(f'Test data accuracy = {accuracy:.5f}\n')



'''
Liver samples
'''
samples = [43, 22, 11, 33, 1, 16]


'''
Create the FACE Explainer
'''
face = FACEExplainer(model)

for sample in samples:
    
    '''
    CLASSIFICATION: FACE Attributions computation
    '''
    
    print(f'Plotting FACE feature attributions for test sample {sample} ground/net={y_test[sample]}/{y_mlp[sample]} ... ', end='')
    
    '''
    And perform the explanation to get the feature attributions
    '''
    face_contrib = face.explain(X_test[sample])
    
    y_face = np.argmax(np.sum(face_contrib, axis=1))
        
    feature_values = scaler.inverse_transform(X_test[sample].reshape(1,-1))[0]
    
    for cl in range(num_outputs):
        plot_bar_contrib('classification', feature_names=feature_names, 
                        contrib_class=face_contrib[cl],
                        pred_class=y_face,
                        real_class=y_test[sample],
                        sample_id=sample, 
                        selected_class=cl, 
                        title=f'Truth/Net/Exp={y_test[sample]}/{y_mlp[sample]}/{y_face} Intercept={face_contrib[cl, 0]:.04f}',
                        show_title=True,
                        max_features=num_inputs,
                        legend='FACE',
                        add_xlabel=True,
                        resize=0.6
                        )

    print('OK')
    
   

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 05:28:18 2024

@author: Carles
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlpxai.explainers.face.kerasmlp import get_face_contrib
# from mlpxai.explainers.face.kerasmlp import hard_sigmoid, hard_tanh

from mlpxai.utils.visualize import plot_bar_contrib

import keras
from keras.models import Model
# from keras.utils.generic_utils import get_custom_objects

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf

import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




'''
CLASSIFICATION
'''


'''
Programa de pruebas de ejecución de la red
'''
# np.seed = 1
# random.seed = 1
# tf.random.set_seed(1)
keras.utils.set_random_seed(1)


# '''
# Incorporamos las funciones hard_sigmoid y hard_tanh
# '''
# get_custom_objects().update({'hard_sigmoid': hard_sigmoid})
# get_custom_objects().update({'hard_tanh': hard_tanh})


ds_name = 'liver'
liver_drinks = 7


if ds_name == 'liver':
    '''
    Liver disorder - Classification
    
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
    
    test_size = 0.15
    validation_split = 0.10
    epochs = 70
    
    use_saved_model_weights = False



'''
Escalamos los datos de entrada con un escalado MinMax y mantenemos el 
escalador para poder utilizar la reversión (scaler.inverse_transform(X))
'''
scaler = MinMaxScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_size,
                                                    shuffle=True,
                                                    stratify=y)

X_test = X_test.to_numpy()

y_train_categorical = to_categorical(y_train, num_outputs).astype(np.float32)


''' 
Creamos y entrenamos el modelo con la definición específica para cada dataset
'''
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='nadam',
              metrics=['accuracy'],
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))


'''
Empezamos el análisis
'''

print('\nCLASSIFICATION OF INDIVIDUALS\n')
print(f'Using dataset: {dataset_name}, {len(y)} samples ({len(y_train)} train / {len(y_test)} test), {num_inputs} features, {num_outputs} classes')

    
print('Training classification model ... ', end='')

my_fit = model.fit(X_train, y_train_categorical,
                    epochs=epochs, validation_split=validation_split, verbose=0)

print('OK')


'''
Realizamos predicciones y evaluamos el resultado
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
samples = [43]


for sample in samples:
    
    '''
    CLASSIFICATION: FACE Attributions computation
    '''
    
    print(f'Plotting FACE feature attributions for test sample {sample} ground/net={y_test[sample]}/{y_mlp[sample]} ... ', end='')
    
    face_contrib = get_face_contrib(X_test[sample], model)
    
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
    
   

# -*- coding: utf-8 -*-

"""
Jose L. Carles - Enrique Carmona - UNED - 2024-2025
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlpxai.explainers.face.kerasface import FACEExplainer

from mlpxai.utils.visualize import plot_bar_contrib, get_str_val

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Dense

import keras
from keras.models import Model



'''
REGRESSION
'''

seed = 1
# np.seed = 1
# random.seed = 1
# tf.random.set_seed(1)
keras.utils.set_random_seed(seed)
       

ds_name = 'delta'


delevators_df = pd.read_csv('sample_data/delta_elevators.csv', delimiter=';')

X = delevators_df.iloc[:, :-1]
y = delevators_df.iloc[:, -1].to_numpy()

feature_names = X.columns
num_inputs =  X.shape[1]
num_outputs = 1

dataset_name = 'Delta Elevators'

input_layer = Input(shape=(num_inputs,))
hidden_layer = Dense(30, activation='relu')(input_layer)
hidden_layer = Dense(20, activation='relu')(hidden_layer)
hidden_layer = Dense(5, activation='relu')(hidden_layer)
output_layer = Dense(num_outputs, activation='linear')(hidden_layer)

test_size = 0.20
validation_split = 0.1 
epochs = 60


use_saved_model_weights = True


'''
Rescale the input data to [0, 1] with a MinMaxScaler. Later we will
use the scaler to transform back the data to the original domain
'''
scaler = MinMaxScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_size,
                                                    shuffle=True,
                                                    random_state=33
                                                    )

X_test = X_test.to_numpy()


''' 
Create and compile the model
'''
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='nadam',
                  loss='mean_squared_error')



'''
As the Delta Elevators dataset is quite big, try to load a 
pretrained model. If it doesn't exist, train and save a new one
'''

weights_file_name = f'{ds_name}_regression_seed_{seed}_epochs_{epochs}.weights.h5'

if use_saved_model_weights:
    if os.path.isfile(weights_file_name):
        model.load_weights(weights_file_name)
        print(
            f'Using pretrained regression model weights from file {weights_file_name}')
    
    else:
        print('Training regression model ... ', end='')
        
        my_fit = model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, verbose=0)
        
        model.save_weights(weights_file_name)
        
        print('OK')
else:
    print(f'Using dataset: {dataset_name}, {len(y)} samples, {num_inputs} features, {num_outputs} output, ({len(y_train)} train / {len(y_test)} test)')

    print('Training regression model ... ', end='')

    my_fit = model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, verbose=0)
    
    print('OK')



'''
Run predictions over the test dataset and show the network performance
'''
print('\nGenerating predictions over test data with MLP ... ', end='')
mlp_predictions = model.predict(X_test, verbose=0)
print('OK')

rmse = np.sqrt(metrics.mean_squared_error(y_test, mlp_predictions))

print(f'MLP results for test data RMSE = {rmse:.5f}')


'''
Create the FACE Explainer
'''
face = FACEExplainer(model)


# Selected sample for Delta
samples = [499, 22, 11, 32, 44, 111]


for sample in samples:

    y_mlp = mlp_predictions[sample][0]
    
    y_mlp_screen = get_str_val(y_mlp)
    
    print(f'\nPlotting FACE regression feature relevance for sample {sample} ... ', end='')
    
    '''
    And perform the explanation to get the feature attributions
    '''
    face_contrib, = face.explain(X_test[sample])
    
    
    y_face = np.sum(face_contrib)
    
    y_face_screen = get_str_val(y_face)
    
    plot_bar_contrib('regression', feature_names, face_contrib, 
      title=f'Truth/Net/Exp={y_test[sample]:.03f}/{y_mlp_screen}/{y_face_screen} Intercept={face_contrib[0]:.03f}',
      legend='FACE',
      add_xlabel=True,
      resize=0.6)
    
    print('OK')
    
    
    
    
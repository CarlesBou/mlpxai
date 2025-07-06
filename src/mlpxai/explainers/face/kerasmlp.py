# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:13:33 2024

@author: Carles
"""

import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout
from keras.engine.input_layer import InputLayer

import pandas as pd


def get_face_contrib(x_sample, model, return_weighted=True):
    if isinstance(x_sample, (pd.core.series.Series)):
        x_sample = x_sample.to_numpy()
        
    _, I_list, H_list, O_list = run_layers(x_sample, model, return_PI_list=True)
        
    contrib = None
    I_index = 0
    
    for layer in model.layers:
        if isinstance(layer, (InputLayer, Dropout)):
            continue
        
        w = layer.get_weights()[0]

        if len(layer.get_weights()) > 1:
            bias = layer.get_weights()[1]
        else:
            bias = None

        w_T_ext = get_transposed_ext(w, bias)

        if I_index == 0:
            contrib = I_list[I_index] @ w_T_ext 
        else:
            contrib =  I_list[I_index] @ w_T_ext @ contrib

        I_index += 1 
        
    if return_weighted:  
        contrib[:, 1:] = contrib[:, 1:] * x_sample

    return contrib[1:] 




def get_layer_type(layer):
  if isinstance(layer, InputLayer):
    return 'InputLayer', '', None
  elif isinstance(layer, Dropout):
    return 'Dropout', '', None
  elif isinstance(layer, Dense): 
    if isinstance(layer.activation, tf.keras.layers.ReLU):
      if layer.activation.negative_slope > 0:
        return 'Dense', 'LeakyReLU', {'negative_slope': layer.activation.negative_slope}
      else:
        return 'Dense', 'ReLU', {'negative_slope': layer.activation.negative_slope} 
    elif isinstance(layer.activation, tf.keras.layers.LeakyReLU):
      return 'Dense', 'LeakyReLU', {'negative_slope': layer.activation.alpha}
    else:
      if hasattr(layer.activation, '__name__'):
        if layer.activation.__name__ == 'linear':
          return 'Dense', 'linear', None
        elif layer.activation.__name__ == 'hard_sigmoid':
          return 'Dense', 'hard_sigmoid', None
        elif layer.activation.__name__ == 'hard_tanh':
          return 'Dense', 'hard_tanh', None
        elif layer.activation.__name__ == 'relu':
          return 'Dense', 'ReLU', 0       
      return 'Dense', 'Unknown', None
  else:
    return 'Unkown', 'Unknown', None
 
  
def print_model_structure(model):    
  print('\nMLP structure')
  for layer in model.layers:
    layer_type, activation_name, activation_params = get_layer_type(layer)
    match layer_type:
      case 'InputLayer':
        print(f'  Layer {layer.name}, InputLayer, Shape {layer.input_shape}')
        continue
      case 'Dropout':
        print(f'  Layer Dropout, rate={layer.rate}')
        continue
      case 'Dense':
        match activation_name:
          case 'LeakyReLU': 
            name = f'Leaky_ReLU, alpha={activation_params["negative_slope"]:.3f}'
          case 'ReLU':
            name = 'ReLU'
          case 'linear':
            name = 'linear'
          case 'hard_sigmoid':
            name = 'hard_sigmoid'
          case 'hard_tanh':
            name = 'hard_tanh'
          case _:
            name = activation_name
            raise ValueError(f'No sé qué es esta función de activación en layer {layer.name}')
      case _:
        raise ValueError(f'No sé qué es este layer {layer}')
  
    if len(layer.get_weights()) > 1: 
      print(f'  Layer {layer.name}, Weights {layer.get_weights()[0].shape}) - Bias {layer.get_weights()[1].shape}, Activation {name}')
    else:
      print(f'  Layer {layer.name}, Weights {layer.get_weights()[0].shape}) - No bias, Activation {name}')

  print()  


'''
Obtenemos la matriz pseudoidentidad para una ReLU para una entrada
de la capa (que tiene unos pesos y bias asodicados)
'''

def get_I_linear(x, w, bias=None):

    x_T_ext = np.hstack((1, x)).reshape(-1,1)
    w_T_ext = get_transposed_ext(w, bias)
    
    H = w_T_ext @ x_T_ext
    
    return np.eye(w_T_ext.shape[0]), np.vstack(([1], H))




def get_I_relu(x, w, bias=None, alpha=0.3):

    x_T_ext = np.hstack((1, x)).reshape(-1,1)
    w_T_ext = get_transposed_ext(w, bias)[1:]
    
    H = w_T_ext @ x_T_ext
    
    mul_pos = H > 0
    mul_pos = mul_pos * 1
    
    mul = mul_pos
    
    if alpha > 0:
        mul_neg = H <= 0
        mul_neg = mul_neg * alpha  
        mul = mul_pos + mul_neg 
          
    I = np.append([1], mul.flatten())
    I = np.diag(I)
    
    return I, I
        

'''
Hard_sigmoid
    np.maximum(0, np.minimum(1, x))
'''
def get_I_hard_sigmoid(x, w, bias=None):
    x_T_ext = np.hstack((1, x)).reshape(-1,1)
    w_T_ext = get_transposed_ext(w, bias)[1:]
  
    H = w_T_ext @ x_T_ext
    
    mul_sup = H > 1
    mul_else = (H > 0) & (H <= 1)
    mul_else = mul_else * 1
    
    mul_sup = np.divide(1, H, where=mul_sup, out=np.zeros(H.shape))
    
    mul = mul_else + mul_sup
        
    I = np.append([1], mul.flatten())
    I = np.diag(I)
    
    return I, np.vstack(([1], H))


'''
Hard_sigmoid
    np.maximum(0, np.minimum(1, x))
'''
def hard_sigmoid_old(x):
    zeros = tf.zeros_like(x)
    ones = tf.ones_like(x)
    return tf.math.maximum(zeros, tf.math.minimum(ones, x))

@tf.custom_gradient
def hard_sigmoid(x):
   def grad(dy):
       return dy * tf.cast(tf.logical_and(x >= 0, x <= 1), dtype=dy.dtype)
   
   return tf.maximum(0.0, tf.minimum(1.0, x)), grad


'''
Hard_tanh
     np.maximum(-1, np.minimum(1, x))
'''
def get_I_hard_tanh(x, w, bias=None):

    x_T_ext = np.hstack((1, x)).reshape(-1,1)
    w_T_ext = get_transposed_ext(w, bias)[1:]
    
    H = w_T_ext @ x_T_ext
    
    mul_plus_one = H > 1
    mul_less_one = H < -1
    mul_else = (H >= -1) & (H <= 1)
    mul_else = mul_else * 1
    
    mul_sup = np.divide(1, H, where=mul_plus_one, out=np.zeros(H.shape))
    
    mul_inf = np.divide(-1, H, where=mul_less_one, out=np.zeros(H.shape))
    
    mul = mul_inf + mul_else + mul_sup
    
    I = np.append([1], mul.flatten())
    I = np.diag(I)
    

    return I, np.vstack(([1], H))


'''
Hard_tanh
    np.maximum(-1, np.minimum(1, x))
'''
def hard_tanh_old(x):
  ones = tf.ones_like(x)
  return tf.math.maximum(-ones, tf.math.minimum(ones, x))

from tensorflow.keras import backend as K


@tf.custom_gradient
def hard_tanh(x):
    def grad(dy):
        # Gradient is 1 in [-1,1], 0 elsewhere
        return dy * tf.cast(tf.logical_and(x >= -1, x <= 1), dtype=dy.dtype)
    
    return K.clip(x, -1.0, 1.0), grad


'''
Obtenemos la matriz pseudoidentidad para una función linean para una entrada
de la capa (que tiene unos pesos y bias asodicados, puede que bias=0)
'''
def get_I_activation(activation, x, w=None, bias=None):

    if isinstance(activation, tf.keras.layers.ReLU):
        alpha = activation.negative_slope
        return get_I_relu(x, w, bias, alpha)

    elif isinstance(activation, tf.keras.layers.LeakyReLU):
        alpha = activation.alpha
        return get_I_relu(x, w, bias, alpha)

    else:
        match activation.__name__:
          case 'relu':
            return get_I_relu(x, w, bias, 0)
          case 'linear':
            return get_I_linear(x, w, bias)
          case 'hard_sigmoid':
            return get_I_hard_sigmoid(x, w, bias)
          case 'hard_tanh':
            return get_I_hard_tanh(x, w, bias)
          case _:   
            assert(f'Unsupported activation function \'{activation.__name__}\'')
    
  

'''
Obtemos la matriz ampliada con el bias incorporado
'''
def get_transposed_ext(w, bias=None):
    if not isinstance(bias, np.ndarray):
        bias = np.zeros(w.shape[1])
      
    bias_T = bias.reshape(-1, 1)
    
    zeros = np.zeros(w.shape[0] + 1)
    zeros[0] = 1
    
    return np.vstack((zeros, np.hstack((bias_T, w.T))))
  

'''
Obtenemos el vector de entrada ampliado
'''
def get_extended_x(x):
  return np.append([1], x).reshape(-1,1)

'''
Ejecución de activación (lineal o ReLU) de una capa
  Transponemos y ampliamos la matriz de pesos incorporando el (posible) bias
  Ampliamos y transponemos el vector de entrada 
  Generamos la matriz de pseudoidentidad (dependiente de la función de activación)
  Operamos la capa
'''
def execute_layer(activation, x, w, bias=None):
    w_T_ext = get_transposed_ext(w, bias)
    x_T_ext = get_extended_x(x)
    
    pseudo_I, H = get_I_activation(activation, x, w, bias)
    
    ret = pseudo_I @ w_T_ext @ x_T_ext
    
    return ret[:,0][1:], pseudo_I, H



        
def run_layers(layer_input, model, return_PI_list=False, return_outputs=False):

    pseudo_I_list = []
    H_list = []
    output_list = []
    
    for layer in model.layers:
        if isinstance(layer, (InputLayer, Dropout)):
            continue
        
        if len(layer.get_weights()) == 2:
            bias = layer.get_weights()[1]
        else:
            bias = None
            
        w = layer.get_weights()[0]
        
        layer_output, pseudo_I, H = execute_layer(layer.activation, layer_input, w, bias) 
        
        pseudo_I_list.append(pseudo_I)
        
        H_list.append(H)
        
        output_list.append(layer_output)
        
        layer_input = layer_output
          
    if return_PI_list:  
        return layer_output, pseudo_I_list, H_list, output_list
    else:
        if return_outputs:
          return layer_output, output_list
        else:
          return layer_output


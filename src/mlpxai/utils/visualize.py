# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 05:28:18 2024

@author: Carles
"""

from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import seaborn as sns



def plot_bar_contrib(mode, *args, **kwargs):
    if mode == 'classification':
        plot_bar_contrib_classification(mode, *args, **kwargs)
    elif mode == 'regression':
        plot_bar_contrib_regression(mode, *args, **kwargs)
        
        
def plot_bar_contrib_classification(mode, feature_names, contrib_class,
                     pred_class=None, real_class=None, sample_id=None, 
                     show='individual', selected_class=None,
                     title='', max_features=12, 
                     reverse_colors=False,
                     show_title=False,
                     bias_name=None,
                     legend=None,
                     add_xlabel=False,
                     resize=1):
    
    sns.set_style("white")

    font_size=35
      
    n_features = min(len(feature_names), max_features)
              
    bias = contrib_class[0]
    contrib_class = contrib_class[1:]
    
    order = np.argsort(abs(contrib_class))[::-1]
    
    contrib_class = contrib_class[order][:n_features]
    feature_names = feature_names[order][:n_features]
      
    if not reverse_colors:
      color = ['b' if c >= 0 else 'r' for c in contrib_class]
    else:
      color = ['r' if c >= 0 else 'b' for c in contrib_class]
      
    if len(feature_names) < 7:
        fig_height = 7.5
    else:
        fig_height = 9
        
    fig, ax = plt.subplots(figsize=(14.5, fig_height))
    
    # Horizontal Bar Plot
    bars = ax.barh(feature_names, abs(contrib_class), color=color, height=0.7)
    
    plt.yticks(fontsize=font_size)
    
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    ax.yaxis.set_tick_params(pad = 10)
    
    for label in ax.get_xticklabels(which='major'):
        label.set(fontsize=font_size - 5)
    
    # Add x, y gridlines
    ax.grid(color ='grey', linestyle ='-.', 
            linewidth = 0.8, alpha = 0.2)
    
    ax.set_ylabel('Feature', fontsize=font_size, labelpad=15)
    
    if add_xlabel:
        ax.set_xlabel('Feature attribution', fontsize=font_size, loc='center', labelpad=15)
    else:
        ax.set_xlabel('                   ', fontsize=font_size, loc='center', labelpad=15)
   
    ax.invert_yaxis()

    labels = [f'{c:.4f}' for c in contrib_class]
    label_width_points = 0
    
    for bars in ax.containers:
        for i, bar in enumerate(bars):
            y = bar.get_y()
            width = bar.get_width()
            height = bar.get_height()
            
            bar_width_points = bar.get_tightbbox().width
            
            if i == 0:
                text = ax.annotate(labels[i], (width/2, y + height/2 + 0.04), 
                                   xytext=(0, 0), ha='center', va='center', 
                                   fontsize=font_size, color='white',
                                   textcoords="offset points")
                label_width_points = text.get_tightbbox().width
            else:
                if bar_width_points >= label_width_points * 1.10:
                    ax.annotate(labels[i], (width/2, y + height/2 + 0.04), 
                                xytext=(0, 0), ha='center', va='center', 
                                fontsize=font_size, color='white',
                                textcoords="offset points")
                else:
                    ax.annotate(labels[i], (width, y + height/2 + 0.04), 
                                xytext=(6, 0), ha='left', va='center', 
                                fontsize=font_size, color='black',
                                textcoords="offset points")

    
    if show_title:
        # Add Plot Title
        if mode == 'classification':
          
            if show == 'individual':
                ax.set_title(title, loc='center', fontsize=font_size)

            elif show == 'mean_class':
                ax.set_title(f'{title}for Class {selected_class} - Intercept={bias:.5f}',
                           loc ='center', fontsize=font_size)
          
        else:
            
          ax.set_title(f'{title}',
                       loc ='center', fontsize=font_size)
      

    if legend is not None:
        plt.annotate(legend, xy=(0.92, 0.3), xycoords='figure fraction', 
                     ha='right', va='center', fontsize=40, color='green',
                     bbox=dict(boxstyle='round,pad=0.2', 
                               facecolor='white',
                               edgecolor='lightgray'))
    
    plt.tight_layout()
        
    fig = plt.gcf()
    current_dpi = fig.get_dpi()
    fig.set_dpi(current_dpi * resize)
    
    plt.show()
        
    return


def plot_bar_contrib_regression(mode, feature_names, contrib_class,
                     pred_class=None, real_class=None, sample_id=None, 
                     show='individual', selected_class=None,
                     title='', max_features=12, 
                     reverse_colors=False,
                     bias_name=None,
                     legend=None,
                     add_xlabel=False,
                     resize=1):
    
    sns.set_style("white")
          
    n_features = min(len(feature_names), max_features)
    
    font_size = 35
            
    bias = contrib_class[0]
    contrib_class = contrib_class[1:]
    
    order = np.argsort(abs(contrib_class))[::-1]
    
    contrib_class = contrib_class[order][:n_features]
    feature_names = feature_names[order][:n_features]
      
    if not reverse_colors:
      color = ['b' if c >= 0 else 'r' for c in contrib_class]
    else:
      color = ['r' if c >= 0 else 'b' for c in contrib_class]
      
    fig, ax = plt.subplots(figsize=(15.5, 9))
    
    # Horizontal Bar Plot
    bars = ax.barh(feature_names, abs(contrib_class), color=color, height=0.7)
    
    plt.yticks(fontsize=font_size)
    
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    ax.yaxis.set_tick_params(pad = 10)
    
    for label in ax.get_xticklabels(which='major'):
        label.set(fontsize=30)
    
    # Add x, y gridlines
    ax.grid(color ='grey', linestyle ='-.', 
            linewidth = 0.8, alpha = 0.2)
    
    ax.set_ylabel('Feature', fontsize=font_size, labelpad=15)
    
    if add_xlabel:
        ax.set_xlabel('Feature attribution', fontsize=font_size, loc='center', labelpad=15)
    else:
        ax.set_xlabel('                   ', fontsize=font_size, loc='center', labelpad=15)
    
    ax.invert_yaxis()

    labels = [get_str_val(c, 4) for c in contrib_class]
    label_width_points = 0
    
    for bars in ax.containers:
        for i, bar in enumerate(bars):
            y = bar.get_y()
            width = bar.get_width()
            height = bar.get_height()
            
            bar_width_points = bar.get_tightbbox().width
            
            if i == 0:
                text = ax.annotate(labels[i], (width/2, y + height/2 + 0.04), 
                                   xytext=(0, 0), ha='center', va='center', 
                                   fontsize=font_size, color='white',
                                   textcoords="offset points")
                label_width_points = text.get_tightbbox().width
            else:
                if bar_width_points >= label_width_points * 1.10:
                    ax.annotate(labels[i], (width/2, y + height/2 + 0.04), 
                                xytext=(0, 0), ha='center', va='center', 
                                fontsize=font_size, color='white',
                                textcoords="offset points")
                else:
                    ax.annotate(labels[i], (width, y + height/2 + 0.04), 
                                xytext=(6, 0), ha='left', va='center', 
                                fontsize=font_size, color='black',
                                textcoords="offset points")
    
    
    # Add Plot Title
    if mode == 'classification':
      
        if show == 'individual':

            ax.set_title(f'{title}for Class={selected_class}, Sample_id={sample_id}, Pred/Real={pred_class}/{real_class} - Intercept={bias:.5f}',
                         loc ='left', fontsize=font_size)
      
        elif show == 'mean_class':
    
            ax.set_title(f'{title}for Class {selected_class} - Intercept={bias:.5f}',
                         loc ='center', fontsize=font_size)
    
    else:
        ax.set_title(f'{title}', loc ='center', fontsize=30)
      
        if legend is not None:
            plt.annotate(legend, xy=(0.92, 0.3), xycoords='figure fraction', 
                         ha='right', va='center', fontsize=40, color='green',
                         bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white',
                                   edgecolor='lightgray'))
      
        plt.tight_layout()

          
    fig = plt.gcf()
    current_dpi = fig.get_dpi()
    fig.set_dpi(current_dpi * resize)

    # Show Plot
    plt.show()
    
    return

    
def plot_MNIST_digit(x_digit, contrib, resize=1):
    
    sns.set(font_scale=1)
      
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.6), clear=True) 

    for x in range(2):

        if x == 0:

            axs[x].axis('off')
            axs[x].imshow(np.reshape(x_digit, (28,28)), 
                              cmap='gray')
            axs[0].set_title('Input')

        else:

            axs[x].imshow(np.reshape(contrib[x-1], (28,28)), 
                              cmap='RdBu_r', 
                              norm=TwoSlopeNorm(vcenter=0),
                              )

            axs[x].axis('off')
            
            axs[x].set_title('FACE')
                
    
    plt.tight_layout()
    
    fig = plt.gcf()
    current_dpi = fig.get_dpi()
    fig.set_dpi(current_dpi * resize)
    
    plt.show()
    
    return
    

def get_str_val(val, decs=3):
    s = f'{val}'
    if decs == 3:
        if val == 0:
            s = '0.000'
        elif abs(val) < 0.001:
            s = f'{val:.02e}'
        else:
            s = f'{val:.03f}'
    elif decs == 4:
        if val == 0:
            s = '0.0000'
        elif abs(val) < 0.0001:
            s = f'{val:.02e}'
        else:
            s = f'{val:.04f}'
    return s
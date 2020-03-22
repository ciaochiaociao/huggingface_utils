from typing import Union

import numpy as np
import seaborn as sns
from ansi.colour import rgb


def color_text(text, rgb_code):
    reset =  '\x1b[0m'
    return rgb.rgb256(*rgb_code) + text + reset

def value2rgb(value):
#     if value < 0:
#         rgb_code = (255/2 + abs(value)/2, abs(value), 255/2 + abs(value)/2)
#     else:
#         rgb_code = (125+value/2, 0, 255/2-value/2)
    if value < 0:
        rgb_code = (255, 255, abs(value))
    else:
        rgb_code = (255, 255-value, 0)
    return rgb_code


def scale(values, input_range, output_range):
    return np.interp(values, input_range, output_range)


def get_legends(value_range, scale_to, step=5):
    min_value, max_value = value_range
    leg_values = np.linspace(min_value, max_value, step)
    scaled_values = scale(leg_values, (min_value, max_value), scale_to)
    
    legends = []
    for leg_value, scaled_value in zip(leg_values, scaled_values):
         legends.append(color_text('{:.2f}'.format(leg_value), value2rgb(scaled_value)))
    return legends


def color_texts(texts, values, use_absolute):
    if use_absolute:
        value_range = (0, 1)
    else:
        value_range = (min(values), max(values))
    scale_to = (-255, 255)
    scaled_values = scale(values, value_range, scale_to)
    result = []
    for text, value in zip(texts, scaled_values):
        rgb = value2rgb(value)
        result.append(color_text(text, rgb))
       
    
    colored = ' '.join(result)
    legends = get_legends(value_range, scale_to)

    colored += ' ({})'.format(' '.join(legends))
        
    if use_absolute:
        colored += ' (min: {:.10f} max: {:.10f})'.format(min(values), max(values))
    
    return colored


def visual_matrix(matrix, labels=None, title=None, **kwargs):

    sns.set()
    ax = sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, **kwargs)
    if title:
        ax.set(title = title)
#     ax.xaxis.tick_top()

    return ax


def get_or_default_config(layer_num, batch_num, head_num, token_num, atn_axis, atns):
    if layer_num is None:
        layer_num = -1  # last layer
    
    batch_size = len(atns[0])
    if batch_size == 1:
        batch_num = 0
    else:
        if batch_num is None:
            raise ValueError('You input an attention with batch size != 1. Please input attentions with batch size 1 or specify the batch_num you want to visualize.')
            
    if head_num is None:
        head_num = 'average'

    if token_num is None:
        token_num = 'average'

    if atn_axis is None:
        atn_axis = 0
        
    return layer_num, batch_num, head_num, token_num, atn_axis


def get_multihead_atn_matrix(atns, layer_num=None, batch_num=None):
    
    
#     layer_num, batch_num = get_or_default_layer_and_batch_num(layer_num, batch_num, atns)
    
    layer = atns[layer_num]

    try:
        multihead_atn_matrix = layer[batch_num].detach().numpy()  # pytorch
    except TypeError:
        multihead_atn_matrix = layer[batch_num].cpu().numpy()  # pytorch
    except AttributeError:
        multihead_atn_matrix = layer[batch_num]  # tensorflow

    return multihead_atn_matrix


def get_atn_matrix_from_mh_matrix(multihead_atn_matrix, head_num):
    # atn_matrix: (sequence_length, sequence_length)       
    try:
        atn_matrix = multihead_atn_matrix[head_num]
    except (IndexError, TypeError):
        # average over heads
        atn_matrix = np.mean(multihead_atn_matrix, axis=0)

    return atn_matrix


def merge_atn_matrix(atn_matrix, mean_over_mat_axis):
    atn_matrix_over_axis: list = np.mean(atn_matrix, axis=mean_over_mat_axis)
    return atn_matrix_over_axis


def matrix2values(matrix, index='average', axis=0):
    
    if index == 'average':
        result_mat = np.mean(matrix, axis=axis)
    elif isinstance(index, int):
        if axis == 0:
            result_mat = matrix[index]
        elif axis == 1:
            result_mat = matrix.T[index]
        else:
            raise ValueError('matrix to values have a wrong axis (0 or 1): ' + str(axis))
    else:
        raise ValueError('matrix to values have a wrong index ("average" or integers): ' + str(index))
    
    return result_mat
        

def get_atn_values(layer_num, batch_num, head_num, token_num, atn_axis, atns):
    layer_num, batch_num, head_num, token_num, atn_axis = get_or_default_config(layer_num, batch_num, head_num, token_num, atn_axis, atns)
    multihead_atn_matrix = get_multihead_atn_matrix(atns, layer_num=layer_num, batch_num=batch_num)
    atn_matrix = get_atn_matrix_from_mh_matrix(multihead_atn_matrix, head_num=head_num)
    atn_values = matrix2values(atn_matrix, index=token_num, axis=atn_axis)
    
    return atn_values


def get_atn_matrix(layer_num, batch_num, head_num, atns):
    layer_num, batch_num, head_num, *_ = get_or_default_config(layer_num, batch_num, head_num, None, None, atns)

    multihead_atn_matrix = get_multihead_atn_matrix(atns, layer_num=layer_num, batch_num=batch_num)
    atn_matrix = get_atn_matrix_from_mh_matrix(multihead_atn_matrix, head_num=head_num)
    return atn_matrix


def visual_atn(labels, atns, layer_num=None, batch_num=None, head_num=None, token_num=None, atn_axis=None,
               use_absolute=False, output=False, **kwargs):
    atn_values = get_atn_values(layer_num, batch_num, head_num, token_num, atn_axis, atns)
    layer_num, batch_num, head_num, token_num, atn_axis = get_or_default_config(layer_num, batch_num, head_num, token_num, atn_axis, atns)

    assert len(labels) == len(atn_values), 'len(labels): {}, len(merged_atn_values): {}'.format(len(labels), len(atn_values))

    colored = color_texts(labels, atn_values, use_absolute)

    try:
        label = labels[token_num]
    except TypeError:
        label = 'ALL_TOKENS'

    print('(layer) {} (batch) {} (head) {} (token_num) {} (token) {} (axis) {}'.format(layer_num, batch_num, head_num, token_num, label, atn_axis))

    if output:
        return colored, atn_values
    else:
        return colored

    
def visual_atn_matrix(labels, atns, layer_num=None, batch_num=None, head_num=None, token_num=None, output=False) -> 'Axes':
    
    atn_matrix = get_atn_matrix(layer_num, batch_num, head_num, atns)
    
    layer_num, batch_num, head_num, token_num, _ = get_or_default_config(layer_num, batch_num, head_num, token_num, None, atns)
    
    title = '(layer) {} (batch) {} (head) {}'.format(layer_num, batch_num, head_num)
    
    if output:
        return visual_matrix(atn_matrix, labels, title=title), atn_matrix
    else:
        return visual_matrix(atn_matrix, labels, title=title)

    
if __name__ == '__main__':

    print(visual_atn(tokens, attentions, use_absolute=True))  # DEFAULT: last layer, average over multi-heads, and average over the 1st axis of self-attention matrix (mean_over_mat_axis=1: np.mean(...,axis=mean_over_mat_axis))

    print(visual_atn(tokens, attentions, layer_num=3, head_num=-1))  # third layer, last head

    visual_atn_matrix(tokens, attentions, layer_num=11)  # second last layer, average over multi-head attention matrix
    plt.show()
    for i in range(12):
        visual_atn_matrix(tokens, attentions, layer_num=-1, head_num=i)  # print attention matrix of every head in the last layer
        plt.show()
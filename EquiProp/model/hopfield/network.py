import random
import numpy as np

from model.function.interaction import SumSeparableFunction
from model.variable.layer import InputLayer, LinearLayer
from model.hopfield.layer import HardSigmoidLayer, ReLU6Layer, ReLUClipLayer
from model.variable.parameter import Bias, DenseWeight, ConvWeight
from model.hopfield.interaction import BiasInteraction, ConvHopfield,  DenseHopfield, ConvMaxPoolHopfield ,  IdentityInteraction
import wandb, json, numbers

def create_layer(shape, activation='hard-sigmoid'):
    """Adds a layer to the network

    Args:
        shape (tuple of ints): shape of the layer
        activation (str, optional): the layer's activation function, either the identity ('linear'), the 'hard-sigmoid', or the `silu'. Default: 'hard-sigmoid'.
    """

    if activation == 'linear': layer = LinearLayer(shape)
    elif activation == 'hard-sigmoid': layer = HardSigmoidLayer(shape)
    elif activation == 'relu6': layer = ReLU6Layer(shape)
    elif activation == 'reluclip': layer = ReLUClipLayer(shape,clip_init=random.uniform(1.,10.))
    elif activation == 'input': layer = InputLayer(shape)
    else: raise ValueError("expected linear, hard-sigmoid, relu6, reluclip or input but got {}".format(activation))

    return layer

def create_edge(layers, interaction_type, indices, gain, shape=None, padding=0,stride=1):
    """Adds an interaction between two layers of the network.

    Adding an interaction also adds the associated parameter (weight or bias)

    Args:
        interaction_type (str): either `bias', `dense', `conv_avg_pool' or `conv_max_pool'
        indices (list of int): indices of layer_pre (the `pre-synaptic' layer) and layer_post (the `post-synaptic' layer)
        gain (float32): the gain (scaling factor) of the param at initialization
        shape (tuple of ints, optional): the shape of the param tensor. Required in the case of convolutional params. Default: None
        padding (int, optional): the padding of the convolution, if applicable. Default: 0
    """

    if interaction_type == "bias":
        layer = layers[indices[0]]
        if shape == None: shape = layer.shape  # if no shape is provided for the bias, we use the layer's shape by default
        param = Bias(shape, gain=gain, device=None)
        interaction = BiasInteraction(layer, param)
    elif interaction_type == "dense":
        layer_pre = layers[indices[0]]
        layer_post = layers[indices[1]]
        param = DenseWeight(layer_pre.shape, layer_post.shape, gain, device=None)
        interaction = DenseHopfield(layer_pre, layer_post, param)
    elif interaction_type == "conv_max_pool":
        layer_pre = layers[indices[0]]
        layer_post = layers[indices[1]]
        param = ConvWeight(shape, gain, device=None)
        interaction = ConvMaxPoolHopfield(layer_pre, layer_post, param, padding,stride)
    elif interaction_type == "identity":
        layer_pre  = layers[indices[0]]
        layer_post = layers[indices[1]]
        param = None
        interaction = IdentityInteraction(layer_pre, layer_post)
    elif interaction_type == "conv":
        layer_pre  = layers[indices[0]]
        layer_post = layers[indices[1]]
        param = ConvWeight(shape, gain, device=None)
        interaction = ConvHopfield(layer_pre, layer_post, param,
                                padding=padding, stride=stride)



    else:
        raise ValueError("expected `bias', `dense', `conv_avg_pool', `conv_max_pool' or `conv_soft_pool' but got {}".format(interaction_type))
    
    return param, interaction



##VGG5
class ConvHopfieldEnergy32(SumSeparableFunction):
    """Energy function of a convolutional Hopfield network (CHN) with 32x32 pixel input images

    The model consists of 5 layers:
        0. input layer has shape (num_inputs, 32, 32)
        1. first hidden layer has shape (num_hidden_1, 16, 16)
        2. second hidden layer has shape (num_hidden_2, 8, 8)
        3. third hidden layer has shape (num_hidden_3, 4, 4)
        4. fourth hidden layer has shape (num_hidden_4, 2, 2)
        5. output layer has shape (num_outputs,)
    
    If num_outputs is None or 0, the model has no output layer.
        
    The first four weight tensors are 3x3 convolutional kernels with padding 1, followed by 2x2 pooling.
    The last weight tensor (if it exists) is dense.
    """

    def __init__(self, num_inputs, num_outputs, num_hiddens_1=128, num_hiddens_2=256, num_hiddens_3=512, num_hiddens_4=512, activation='hard-sigmoid', interaction_type='conv_max_pool', weight_gains=[0.5, 0.5, 0.5, 0.5, 0.5]):
        """Creates an instance of a convolutional Hopfield network 32x32 (CHN32).

        Args:
            num_inputs (int): number of input filters
            num_outputs (int or None): number of output units
            num_hiddens_1 (int, optional): number of filters in the first hidden layer. Default: 128
            num_hiddens_2 (int, optional): number of filters in the second hidden layer. Default: 256
            num_hiddens_3 (int, optional): number of filters in the third hidden layer. Default: 512
            num_hiddens_4 (int, optional): number of filters in the fourth hidden layer. Default: 512
            activation (str, optional): activation function used for the hidden layers. Default: 'hard-sigmoid'
            interaction_type (str, optional): the type of pooling operation used. Default: 'conv_max_pool'
            weight_gains (list of float, optional): the numbers used to scale the weights, layer-wise. Default: gain=0.5 for each weight
        """

        self._size = [num_inputs, num_hiddens_1, num_hiddens_2, num_hiddens_3, num_hiddens_4, num_outputs]
        self._activation = activation
        self._interaction_type = interaction_type
        self._weight_gains = weight_gains

        # layers of the network
        layer_shapes = [(num_inputs, 32, 32), (num_hiddens_1, 16, 16), (num_hiddens_2, 8, 8), (num_hiddens_3, 4, 4), (num_hiddens_4, 2, 2), (num_outputs,)]
        activations = ['input', activation, activation, activation, activation, 'linear']

        # biases of the network
        bias_shapes = [(num_hiddens_1,), (num_hiddens_2,), (num_hiddens_3,), (num_hiddens_4,), (num_outputs,)]
        bias_gains = [0.5/np.sqrt(num_inputs*3*3), 0.5/np.sqrt(num_hiddens_1*3*3), 0.5/np.sqrt(num_hiddens_2*3*3), 0.5/np.sqrt(num_hiddens_3*3*3), 0.5/np.sqrt(num_hiddens_4*2*2)]

        # weights of the network
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        weight_shapes = [(num_hiddens_1, num_inputs, 3, 3), (num_hiddens_2, num_hiddens_1, 3, 3), (num_hiddens_3, num_hiddens_2, 3, 3), (num_hiddens_4, num_hiddens_3, 3, 3), (num_hiddens_4, 2, 2, num_outputs)]
        weight_types = [interaction_type, interaction_type, interaction_type, interaction_type, 'dense']
        paddings = [1, 1, 1, 1, None]

        if num_outputs == None or num_outputs == 0:
            layer_shapes = layer_shapes[:-1]
            activations = activations[:-1]
            bias_shapes = bias_shapes[:-1]
            bias_gains = bias_gains[:-1]
            edges = edges[:-1]
            weight_types = weight_types[:-1]
            # weight_gains = weight_gains[:-1]  # FIXME: careful not to add this line or RO breaks
            weight_shapes = weight_shapes[:-1]
            paddings = paddings[:-1]

        # create the layers, biases and weights
        layers = [create_layer(shape, activation) for shape, activation in zip(layer_shapes, activations)]

        biases = [Bias(shape, gain=gain, device=None) for shape, gain in zip(bias_shapes, bias_gains)]  # the bias has the same shape as the layer
        bias_interactions = [BiasInteraction(layer, bias) for layer, bias in zip(layers[1:], biases)]

        params = biases
        interactions = bias_interactions

        for indices, weight_type, gain, shape, padding, in zip(edges, weight_types, weight_gains, weight_shapes, paddings):
            param, interaction = create_edge(layers, weight_type, indices, gain, shape, padding)
            params.append(param)
            interactions.append(interaction)

        # creates an instance of a SumSeparableFunction
        SumSeparableFunction.__init__(self, layers, params, interactions)

    def __str__(self):
        return 'ConvHopfieldEnergy32 -- size={}, activation={}, pooling={}, gains={}'.format(self._size, self._activation, self._interaction_type, self._weight_gains)
    


def _json_safe(obj):
    """Convert tuples / numpy / tensors → JSON‑friendly Python."""
    if isinstance(obj, (list, dict, str, bool)) or obj is None:
        return obj
    if isinstance(obj, numbers.Number):
        return float(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)

# Hopfield-Resnet 13
class ConvHopfieldResEnergy32(SumSeparableFunction):
    """Energy function of a convolutional Hopfield network (CHN) with 32x32 pixel input images

    The model consists of 9 neuronal states and 13 weight tensors:
    """

    def __init__(self, num_inputs, num_outputs, num_hiddens_1=128, num_hiddens_2=256, num_hiddens_3=512, num_hiddens_4=512, activation='hard-sigmoid', interaction_type='conv', weight_gains=[0.5, 0.5, 0.5, 0.5, 0.5]):
        """Creates an instance of a convolutional Hopfield network 32x32 (CHN32).

        Args:
            num_inputs (int): number of input filters
            num_outputs (int or None): number of output units
            num_hiddens_1 (int, optional): number of filters in the first hidden layer. Default: 128
            num_hiddens_2 (int, optional): number of filters in the second hidden layer. Default: 256
            num_hiddens_3 (int, optional): number of filters in the third hidden layer. Default: 512
            num_hiddens_4 (int, optional): number of filters in the fourth hidden layer. Default: 512
            activation (str, optional): activation function used for the hidden layers. Default: 'hard-sigmoid'
            interaction_type (str, optional): the type of pooling operation used. Default: 'conv_max_pool'
            weight_gains (list of float, optional): the numbers used to scale the weights, layer-wise. Default: gain=0.5 for each weight
        """

        self._size = [num_inputs, num_hiddens_1, num_hiddens_2, num_hiddens_3, num_hiddens_4, num_outputs]
        self._activation = activation
        self._interaction_type = interaction_type
        self._weight_gains = weight_gains

        # 1) Neuronal state shapes
        layer_shapes = [
            (num_inputs,    32, 32),   # 0  input
            (num_hiddens_1, 32, 32),   # 1  h1a
            (num_hiddens_1, 16, 16),   # 2  h1b
            (num_hiddens_2, 16, 16),   # 3  h2a
            (num_hiddens_2,  8,  8),   # 4  h2b
            (num_hiddens_3,  8,  8),   # 5  h3a
            (num_hiddens_3,  4,  4),   # 6  h3b
            (num_hiddens_4,  4,  4),   # 7  h4a  
            (num_hiddens_4,  2,  2),   # 8  h4b  
            (num_outputs,)             # 9  out
        ]

        # 2) activations
        activations = ['input'] + [activation] * 8 + ['linear'] #activation for all neuronal states except input and output

        # 3) biases
        bias_shapes = [
            (num_hiddens_1,), (num_hiddens_1, ),
            (num_hiddens_2,), (num_hiddens_2, ),
            (num_hiddens_3,), (num_hiddens_3, ),
            (num_hiddens_4,), (num_hiddens_4, ),
            (num_outputs,)
        ]
        bias_gains = [
            0.5 / np.sqrt(num_inputs    * 3 * 3),
            0.5 / np.sqrt(num_hiddens_1 * 3 * 3),
            0.5 / np.sqrt(num_hiddens_1 * 3 * 3),
            0.5 / np.sqrt(num_hiddens_2 * 3 * 3),
            0.5 / np.sqrt(num_hiddens_2 * 3 * 3),
            0.5 / np.sqrt(num_hiddens_3 * 3 * 3),
            0.5 / np.sqrt(num_hiddens_3 * 3 * 3),
            0.5 / np.sqrt(num_hiddens_4 * 3 * 3),
            0.5 / np.sqrt(num_hiddens_4 * 3 * 3),
        ]

        # 4) edges  (13 total)
        edges = [
            (0,1), (1,2), (0,2),                 # block‑1
            (2,3), (2,4), (3,4),                 # block‑2
            (4,5), (4,6), (5,6),                 # block‑3
            (6,7), (6,8), (7,8),                 # block‑4
            (8,9)                                # dense head
        ]

        # 5) interaction types
        weight_types = [
            'conv','conv','conv',                # block‑1
            'conv','conv','conv',                # block‑2
            'conv','conv','conv',                # block‑3
            'conv','conv','conv',                # block‑4 
            'dense'                              # dense head
        ]

        # 6) weight shapes
        weight_shapes = [
            (num_hiddens_1, num_inputs,    3, 3),   # 0→1
            (num_hiddens_1, num_hiddens_1, 3, 3),   # 1→2
            (num_hiddens_1, num_inputs,    1, 1),   # 0→2 skip

            (num_hiddens_2, num_hiddens_1, 3, 3),   # 2→3
            (num_hiddens_2, num_hiddens_1, 1, 1),   # 2→4 skip
            (num_hiddens_2, num_hiddens_2, 3, 3),   # 3→4

            (num_hiddens_3, num_hiddens_2, 3, 3),   # 4→5
            (num_hiddens_3, num_hiddens_2, 1, 1),   # 4→6 skip
            (num_hiddens_3, num_hiddens_3, 3, 3),   # 5→6

            (num_hiddens_4, num_hiddens_3, 3, 3),   # 6→7  
            (num_hiddens_4, num_hiddens_3, 1, 1),   # 6→8  skip (
            (num_hiddens_4, num_hiddens_4, 3, 3),   # 7→8  (s2)

            (num_hiddens_4, 1, 1, num_outputs),     # 8→9  dense
        ]

        # 7) paddings
        paddings = [1,1,0, 1,0,1, 1,0,1, 1,0,1, None]

        # 8) strides
        strides  = [
            1,2,2,          # block‑1
            1,2,2,          # block‑2
            1,2,2,          # block‑3
            1,2,2,          # block‑4
            None            # dense
        ]
        if num_outputs == None or num_outputs == 0:
            layer_shapes = layer_shapes[:-1]
            activations = activations[:-1]
            bias_shapes = bias_shapes[:-1]
            bias_gains = bias_gains[:-1]
            edges = edges[:-1]
            weight_types = weight_types[:-1]
            
            weight_shapes = weight_shapes[:-1]
            paddings = paddings[:-1]

        # create the layers, biases and weights
        layers = [create_layer(shape, activation) for shape, activation in zip(layer_shapes, activations)]

        biases = [Bias(shape, gain=gain, device=None) for shape, gain in zip(bias_shapes, bias_gains)]  # the bias has the same shape as the layer
        bias_interactions = [BiasInteraction(layer, bias) for layer, bias in zip(layers[1:], biases)]

        params = biases
        interactions = bias_interactions

        for indices, weight_type, gain, shape, padding,stride in zip(edges, weight_types, weight_gains, weight_shapes, paddings,strides):
            param, interaction = create_edge(layers, weight_type, indices, gain, shape, padding,stride)
            params.append(param)
            interactions.append(interaction)

        # creates an instance of a SumSeparableFunction
        SumSeparableFunction.__init__(self, layers, params, interactions)
        if wandb.run is not None and "architecture" not in wandb.config:
                    arch_cfg = {
                        "num_inputs":  num_inputs,
                        "num_outputs": num_outputs,
                        "hidden_sizes":[num_hiddens_1, num_hiddens_2,
                                        num_hiddens_3, num_hiddens_4],
                        "activation":  activation,
                        "layer_shapes":[_json_safe(s) for s in layer_shapes],
                        "edges":       [_json_safe(e) for e in edges],
                        "weight_types":weight_types,
                        "paddings":    paddings,
                        "strides":     strides,
                        "weight_shapes":[_json_safe(s) for s in weight_shapes],
                        "weight_gains":_json_safe(weight_gains),
                    }
                    wandb.config.update({"architecture": arch_cfg},
                                        allow_val_change=True)

    def __str__(self):
        return 'ConvHopfieldResnet -- size={}, activation={}, pooling={}, gains={}'.format(self._size, self._activation, self._interaction_type, self._weight_gains)


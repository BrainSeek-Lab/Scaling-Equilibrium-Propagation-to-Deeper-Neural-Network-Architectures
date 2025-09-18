from abc import ABC
import torch
import torch.nn.functional as F
from model.variable.layer import Layer

class HardSigmoidLayer(Layer):
    """
    Class used to implement a layer with a hard-sigmoid activation function

    Methods
    -------
    activate():
        Returns the value of the layer's state, clamped between 0 and 1
    """

    def activate(self):
        """Returns the value of the layer's state, clamped between 0 and 1"""
        return self._state.clamp(min=0., max=1.)


class ReLULayer(Layer):
    """
    Class used to implement a layer with a relu activation function

    Methods
    -------
    activate():
        Applies the relu function to the layer's state and returns the result
    """

    def activate(self):
        """Returns the relu function applied to the layer's state"""
        return torch.relu(self._state)
class ReLU6Layer(Layer):
    """
    Class used to implement a layer with a relu6 activation function 

    Methods
    -------
    activate():
        Applies the logistic function to the layer's state and returns the result
    """

    def activate(self):
        """Returns the logistic function applied to the layer's state"""
        return F.relu6(self._state)
class ReLUClipLayer(Layer):
    def __init__(self, shape, clip_init=6.0, learnable=False):
        super().__init__(shape)

        # Initialize clip value
        clip_val = torch.tensor(float(clip_init))

        if learnable:
            # Make it a learnable parameter
            self.clip_value = torch.nn.Parameter(clip_val)
        else:
            # Just a plain tensor (not registered)
            self.clip_value = clip_val

        self.learnable = learnable

    def activate(self):
        return torch.clamp(torch.relu(self._state), 0.0, self.clip_value)

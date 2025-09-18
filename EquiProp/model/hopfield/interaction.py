import torch
import torch.nn.functional as F

from model.function.interaction import Function 

class IdentityInteraction(Function):
    """Fixed skip: energy  E = -〈z_pre , z_post〉."""

    def __init__(self, layer_pre, layer_post):
        self._layer_pre  = layer_pre
        self._layer_post = layer_post
        super().__init__([layer_pre, layer_post], [])

    # ---------- energy ----------
    def eval(self):
        z_pre  = self._layer_pre.state          # shape (B,C,H,W) or (B,C)
        z_post = self._layer_post.state
        # sum over every dim except batch
        reduce_dims = tuple(range(1, z_pre.dim()))
        return -(z_pre * z_post).sum(dim=reduce_dims)

    # ---------- gradients w.r.t. layers ----------
    def grad_layer_fn(self, layer):
        if layer is self._layer_pre:
            return lambda: -self._layer_post.state
        elif layer is self._layer_post:
            return lambda: -self._layer_pre.state
        else:
            raise ValueError("Layer not part of this interaction")

    # ---------- gradients w.r.t. parameters ----------
    def grad_param_fn(self, param):
        raise RuntimeError("IdentityInteraction has no parameters")


class BiasInteraction(Function):
    """Interaction of the Bias

    A bias interaction is defined between a layer and its corresponding bias variable

    Attributes
    ----------
    _layer (Layer): the layer involved in the interaction
    _bias (Bias): the layer's bias
    """

    def __init__(self, layer, bias):
        """Creates an instance of BiasInteraction

        Args:
            layer (Layer): the layer involved in the interaction
            bias (Bias): the layer's bias
        """

        self._layer = layer
        self._bias = bias

        Function.__init__(self, [layer], [bias])

    def eval(self):
        """Energy function of the bias interaction

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        # FIXME: we need to broadcast the bias tensor to the same shape as the layer tensor, to make sure that the correct dimensions are multiplied together.

        bias = self._bias.get().unsqueeze(0)
        while len(bias.shape) < len(self._layer.state.shape): bias = bias.unsqueeze(-1)

        return - self._layer.state.mul(bias).flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        """Overrides the default implementation of Function"""
        dictionary = {self._layer: self._grad_layer}
        return dictionary[layer]

    def grad_param_fn(self, param):
        """Overrides the default implementation of Function"""
        dictionary = {self._bias: self._grad_bias}
        return dictionary[param]

    def _grad_layer(self):
        """Returns the interaction's gradient wrt the layer

        Returns:
            Tensor of shape (batch_size, layer_shape) and type float32: the gradient
        """

        # FIXME: we need to broadcast the bias tensor to the same shape as the layer tensor, to make sure that the correct dimensions are added together.
        grad = - self._bias.get().unsqueeze(0)
        while len(grad.shape) < len(self._layer.state.shape): grad = grad.unsqueeze(-1)
        return grad

    def _grad_bias(self):
        """Returns the interaction's gradient wrt the bias"""

        # FIXME: we need to broadcast the bias tensor to the same shape as the layer tensor, to make sure that the correct dimensions are added together.

        grad = - self._layer.state.mean(dim=0)

        if len(grad.shape) > len(self._bias.shape):
            dims = tuple(range(len(self._bias.shape), len(grad.shape)))
            grad = grad.sum(dim=dims)
            
        return grad


class DenseHopfield(Function):
    """Dense ('fully connected') interaction between two layers

    A dense interaction is defined between three variables: two adjacent layers, and the corresponding weight tensor between the two.
    
    Attributes
    ----------
    _layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_pre_shape). Type is float32.
    _layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_post_shape). Type is float32.
    _weight (DenseWeight): weight tensor between layer_pre and layer_post. Tensor of shape (layer_pre_shape, layer_post_shape). Type is float32.
    """

    def __init__(self, layer_pre, layer_post, dense_weight):
        """Creates an instance of DenseHopfield

        Args:
            layer_pre (Layer): pre-synaptic layer
            layer_post (Layer): post-synaptic layer
            dense_weight (DenseWeight): the dense weights between the pre- and post-synaptic layer
        """

        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = dense_weight

        Function.__init__(self, [layer_pre, layer_post], [dense_weight])

    def eval(self):
        """Returns the energy of a dense interaction.
        
        Example:
            - layer_pre is of shape (16, 1, 28, 28), i.e. batch_size is 16, with 1 channel of 28 by 28 (e.g. input tensor for MNIST)
            - layer_post is of shape (16, 2048), i.e. batch_size is 16, with 2048 units
            - weight is of shape (1, 28, 28, 2048)
        pre * W is the tensor product of pre and W over the dimensions (1, 28, 28). The result is a tensor of shape (16, 2048).
        
        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)  # number of dimensions involved in the tensor product layer_pre * weight
        return - torch.tensordot(layer_pre, self._weight.get(), dims=dims_pre).mul(layer_post).flatten(start_dim=1).sum(dim=1)  # Hebbian term: layer_pre * weight * layer_post

    def grad_layer_fn(self, layer):
        """Overrides the default implementation of Function"""
        dictionary = {
            self._layer_pre: self._grad_pre,
            self._layer_post: self._grad_post
            }
        return dictionary[layer]

    def grad_param_fn(self, param):
        """Overrides the default implementation of Function"""
        dictionary = {self._weight: self._grad_weight}
        return dictionary[param]

    def _grad_pre(self):
        """Returns the gradient of the energy function wrt the pre-synaptic layer.

        This is the usual - weight * layer_post

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the gradient wrt layer_pre
        """

        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)  # number of dimensions involved in the tensor product
        dim_weight = len(self._weight.shape)
        permutation = tuple(range(dims_pre, dim_weight)) + tuple(range(dims_pre))
        return - torch.tensordot(layer_post, self._weight.get().permute(permutation), dims=dims_post)

    def _grad_post(self):
        """Returns the gradient of the energy function wrt the post-synaptic layer.

        This is the usual - layer_pre * weight

        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the gradient wrt layer_post
        """

        layer_pre = self._layer_pre.state
        dims_pre = len(self._layer_pre.shape)  # number of dimensions involved in the tensor product
        return - torch.tensordot(layer_pre, self._weight.get(), dims=dims_pre)

    def _grad_weight(self):
        """Returns the gradient of the energy function wrt the weight.

        This is the usual Hebbian term, dE/dtheta = - layer_pre^T * layer_post

        Returns:
            Tensor of shape weight_shape and type float32: the gradient wrt the weights
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state
        batch_size = layer_pre.shape[0]
        return - torch.tensordot(layer_pre, layer_post, dims=([0], [0])) / batch_size  # we divide by batch size because we want the mean gradient over the mini-batch


class ConvHopfield(Function):
    """Pure convolutional interaction between two layers (no pooling)."""

    def __init__(self, layer_pre, layer_post, conv_weight, padding=0, stride=1):
        self._layer_pre  = layer_pre
        self._layer_post = layer_post
        self._weight     = conv_weight
        self._padding    = padding
        self._stride     = stride

        super().__init__([layer_pre, layer_post], [conv_weight])

    def eval(self):
        """E = -⟨ conv(z_pre, W), z_post ⟩  (sum over C,H,W)."""
        z_pre  = self._layer_pre.state
        z_post = self._layer_post.state
        conv   = F.conv2d(z_pre, self._weight.get(),
                          padding=self._padding, stride=self._stride)
        return -(conv * z_post).sum(dim=(1, 2, 3))       
    def grad_layer_fn(self, layer):
        return {self._layer_pre:  self._grad_pre,
                self._layer_post: self._grad_post}[layer]

    def grad_param_fn(self, param):
        return {self._weight: self._grad_weight}[param]

    def _grad_pre(self):
        z_post = self._layer_post.state
        W      = self._weight.get()

        # choose output_padding so that output size matches layer_pre
        H_in = self._layer_pre.state.shape[2]
        out_h = (z_post.shape[2] - 1) * self._stride \
                - 2 * self._padding + W.shape[2]
        output_pad = H_in - out_h          # 0 or 1 in our case

        return -F.conv_transpose2d(
            z_post, W,
            padding=self._padding,
            stride=self._stride,
            output_padding=int(output_pad)  # <‑‑ NEW
        )

    def _grad_post(self):
        z_pre = self._layer_pre.state
        W     = self._weight.get()
        return -F.conv2d(z_pre, W,
                         padding=self._padding, stride=self._stride)
    def _grad_weight(self):
        layer_pre  = self._layer_pre.state    # shape: (B, in_c, H, W)
        layer_post = self._layer_post.state   # shape: (B, out_c, H', W')
        W_shape    = self._weight.get().shape # expected: (out_c, in_c, kH, kW)

        return - F.grad.conv2d_weight(
            input=layer_pre,
            weight_size=W_shape,
            grad_output=layer_post,
            padding=self._padding,
            stride=self._stride
        ) / layer_pre.shape[0]  # average over batch


class ConvMaxPoolHopfield(Function):
    """Convolutional interaction between two layers, with 2*2 max pooling.

    Attributes
    ----------
    _layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
    _layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
    _weight (ConvWeight): convolutional weight tensor between layer_pre and layer_post. Type is float32.
    _padding (int): padding of the convolution.
    """

    def __init__(self, layer_pre, layer_post, conv_weight, padding=0,stride=1):
        """Creates an instance of ConvMaxPoolHopfield

        Args:
            layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
            layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
            conv_weight (ConvWeight): convolutional weights between layer_pre and layer_post.
            padding (int, optional): padding of the convolution. Default: 0
        """

        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = conv_weight
        self._padding = padding
        self._stride=stride

        Function.__init__(self, [layer_pre, layer_post], [conv_weight])

    def eval(self):
        """Returns the energy of a convolutional interaction with max pooling.

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state

        return - F.max_pool2d(F.conv2d(layer_pre, self._weight.get(), padding=self._padding,stride=self._stride), 2).mul(layer_post).sum(dim=(3,2,1))

    def grad_layer_fn(self, layer):
        """Overrides the default implementation of Function"""
        dictionary = {
            self._layer_pre: self._grad_pre,
            self._layer_post: self._grad_post
            }
        return dictionary[layer]

    def grad_param_fn(self, param):
        """Overrides the default implementation of Function"""
        dictionary = {self._weight: self._grad_weight}
        return dictionary[param]
    def _grad_pre(self):
            layer_pre = self._layer_pre.state
            
            # Calculate the output size of the convolution before pooling
            # This is the target output_size for max_unpool2d
            H_in, W_in = layer_pre.shape[2], layer_pre.shape[3]
            K_H, K_W = self._weight.get().shape[2], self._weight.get().shape[3]
            conv_out_H = (H_in + 2 * self._padding - K_H) // self._stride + 1
            conv_out_W = (W_in + 2 * self._padding - K_W) // self._stride + 1
            num_output_channels_conv = self._weight.get().shape[0]

            # This unpool_output_size should be the size of the tensor *before* max_pool2d
            unpool_output_size = (layer_pre.shape[0], num_output_channels_conv, conv_out_H, conv_out_W) # e.g. (128, 256, 8, 8) for (1,2) edge

            # Perform the forward convolution to get the pre-pooled tensor and its indices
            conv_out = F.conv2d(layer_pre, self._weight.get(), padding=self._padding, stride=self._stride)
            _, indices = F.max_pool2d(conv_out, 2, return_indices=True) # indices are for pooling from conv_out (e.g. 8x8) to pooled (e.g. 4x4)

            # layer_post is now the gradient coming from the output of the pooled layer
            # If layer_shapes are corrected, layer_post (e.g. Layer 2) will be (batch_size, C2, 4, 4)
            layer_post = self._layer_post.state 

            # Unpooling operation: unpool the gradient from layer_post using the indices
            # layer_post (e.g. 4x4) is the 'input' to max_unpool2d
            # indices are for 8x8 -> 4x4 pooling
            # output_size is 8x8
            unpooled_grad = F.max_unpool2d(layer_post, indices, 2, output_size=unpool_output_size)
            
            return - F.conv_transpose2d(unpooled_grad, self._weight.get(), padding=self._padding, stride=self._stride)
    def _grad_post(self):
        """Returns the gradient of the energy function wrt the post-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the gradient wrt layer_post
        """

        layer_pre = self._layer_pre.state
        return - F.max_pool2d(F.conv2d(layer_pre, self._weight.get(), padding=self._padding,stride=self._stride), 2)

    def _grad_weight(self):
        """Returns the gradient of the energy function wrt the weight."""
        layer_pre = self._layer_pre.state
        W_shape   = self._weight.get().shape

        # Forward pass to get pooling indices
        conv_out, indices = F.max_pool2d(
            F.conv2d(layer_pre, self._weight.get(), padding=self._padding, stride=self._stride),
            2,
            return_indices=True
        )

        # Unpool the layer_post using those indices
        layer_post = self._layer_post.state
        unpooled_post = F.max_unpool2d(layer_post, indices, 2)

        # Compute true gradient w.r.t. conv weights
        return -F.grad.conv2d_weight(
            input=layer_pre,
            weight_size=W_shape,
            grad_output=unpooled_post,
            padding=self._padding,
            stride=self._stride
        ) / layer_pre.shape[0]  # mean over batch

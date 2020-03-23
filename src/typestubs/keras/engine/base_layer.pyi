"""
This type stub file was generated by pyright.
"""

import threading
from .. import backend as K
from typing import Any, Optional

"""Contains the base Layer class, from which all layers inherit.
"""
_DISABLE_TRACKING = threading.local()
def disable_tracking(func):
    ...

class Layer(object):
    """Abstract base layer class.

    # Properties
        input, output: Input/output tensor(s). Note that if the layer
            is used more than once (shared layer), this is ill-defined
            and will raise an exception. In such cases, use
            `layer.get_input_at(node_index)`.
        input_mask, output_mask: Mask tensors. Same caveats apply as
            input, output.
        input_shape: Shape tuple. Provided for convenience, but note
            that there may be cases in which this attribute is
            ill-defined (e.g. a shared layer with multiple input
            shapes), in which case requesting `input_shape` will raise
            an Exception. Prefer using
            `layer.get_input_shape_at(node_index)`.
        input_spec: List of InputSpec class instances
            each entry describes one required input:
                - ndim
                - dtype
            A layer with `n` input tensors must have
            an `input_spec` of length `n`.
        name: String, must be unique within a model.
        non_trainable_weights: List of variables.
        output_shape: Shape tuple. See `input_shape`.
        stateful: Boolean indicating whether the layer carries
            additional non-weight state. Used in, for instance, RNN
            cells to carry information between batches.
        supports_masking: Boolean indicator of whether the layer
            supports masking, typically for unused timesteps in a
            sequence.
        trainable: Boolean, whether the layer weights
            will be updated during training.
        trainable_weights: List of variables.
        uses_learning_phase: Whether any operation
            of the layer uses `K.in_training_phase()`
            or `K.in_test_phase()`.
        weights: The concatenation of the lists trainable_weights and
            non_trainable_weights (in this order).
        dtype:  Default dtype of the layers's weights.


    # Methods
        call(x, mask=None): Where the layer's logic lives.
        __call__(x, mask=None): Wrapper around the layer logic (`call`).
            If x is a Keras tensor:
                - Connect current layer with last layer from tensor:
                    `self._add_inbound_node(last_layer)`
                - Add layer to tensor history
            If layer is not built:
                - Build from x._keras_shape
        compute_mask(x, mask)
        compute_output_shape(input_shape)
        count_params()
        get_config()
        get_input_at(node_index)
        get_input_mask_at(node_index)
        get_input_shape_at(node_index)
        get_output_at(node_index)
        get_output_mask_at(node_index)
        get_output_shape_at(node_index)
        get_weights()
        set_weights(weights)

    # Class Methods
        from_config(config)

    # Internal methods:
        _add_inbound_node(layer, index=0)
        assert_input_compatibility()
        build(input_shape)
    """
    def __init__(self, **kwargs):
        self.input_spec = ...
        self.supports_masking = ...
        self.stateful = ...
        self.name = ...
        self.trainable = ...
        self.dtype = ...
    
    @staticmethod
    def _node_key(layer, node_index):
        """Converts a layer and its index to a unique (immutable type) name.

        This function is used internally with `self._network_nodes`.

        # Arguments
            layer: The layer.
            node_index: The layer's position (e.g. via enumerate) in a list of
                nodes.

        # Returns
            The unique name.
        """
        ...
    
    @property
    def losses(self):
        ...
    
    @property
    def updates(self):
        ...
    
    @property
    def built(self):
        ...
    
    @built.setter
    def built(self, value):
        ...
    
    @property
    def trainable_weights(self):
        ...
    
    @trainable_weights.setter
    def trainable_weights(self, weights):
        ...
    
    @property
    def non_trainable_weights(self):
        ...
    
    @non_trainable_weights.setter
    def non_trainable_weights(self, weights):
        ...
    
    def add_weight(self, name: Optional[Any] = ..., shape: Optional[Any] = ..., dtype: Optional[Any] = ..., initializer: Optional[Any] = ..., regularizer: Optional[Any] = ..., trainable: bool = ..., constraint: Optional[Any] = ...):
        """Adds a weight variable to the layer.

        # Arguments
            name: String, the name for the weight variable.
            shape: The shape tuple of the weight.
            dtype: The dtype of the weight.
            initializer: An Initializer instance (callable).
            regularizer: An optional Regularizer instance.
            trainable: A boolean, whether the weight should
                be trained via backprop or not (assuming
                that the layer itself is also trainable).
            constraint: An optional Constraint instance.

        # Returns
            The created weight variable.
        """
        ...
    
    def assert_input_compatibility(self, inputs):
        """Checks compatibility between the layer and provided inputs.

        This checks that the tensor(s) `input`
        verify the input assumptions of the layer
        (if any). If not, exceptions are raised.

        # Arguments
            inputs: input tensor or list of input tensors.

        # Raises
            ValueError: in case of mismatch between
                the provided inputs and the expectations of the layer.
        """
        ...
    
    def call(self, inputs, **kwargs):
        """This is where the layer's logic lives.

        # Arguments
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        # Returns
            A tensor or list/tuple of tensors.
        """
        ...
    
    @K.symbolic
    def __call__(self, inputs, **kwargs):
        """Wrapper around self.call(), for handling internal references.

        If a Keras tensor is passed:
            - We call self._add_inbound_node().
            - If necessary, we `build` the layer to match
                the _keras_shape of the input(s).
            - We update the _keras_shape of every input tensor with
                its new shape (obtained via self.compute_output_shape).
                This is done as part of _add_inbound_node().
            - We update the _keras_history of the output tensor(s)
                with the current layer.
                This is done as part of _add_inbound_node().

        # Arguments
            inputs: Can be a tensor or list/tuple of tensors.
            **kwargs: Additional keyword arguments to be passed to `call()`.

        # Returns
            Output of the layer's `call` method.

        # Raises
            ValueError: in case the layer is missing shape information
                for its `build` call.
        """
        ...
    
    def _add_inbound_node(self, input_tensors, output_tensors, input_masks, output_masks, input_shapes, output_shapes, arguments: Optional[Any] = ...):
        """Internal method to create an inbound node for the layer.

        # Arguments
            input_tensors: list of input tensors.
            output_tensors: list of output tensors.
            input_masks: list of input masks (a mask can be a tensor, or None).
            output_masks: list of output masks
                (a mask can be a tensor, or None).
            input_shapes: list of input shape tuples.
            output_shapes: list of output shape tuples.
            arguments: dictionary of keyword arguments that were passed to the
                `call` method of the layer at the call that created the node.
        """
        ...
    
    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Assumes that the layer will be built
        to match that input shape provided.

        # Arguments
            input_shape: Shape tuple (tuple of integers)
                or list of shape tuples (one per output tensor of the layer).
                Shape tuples can include None for free dimensions,
                instead of an integer.

        # Returns
            An output shape tuple.
        """
        ...
    
    def compute_mask(self, inputs, mask: Optional[Any] = ...):
        """Computes an output mask tensor.

        # Arguments
            inputs: Tensor or list of tensors.
            mask: Tensor or list of tensors.

        # Returns
            None or a tensor (or list of tensors,
                one per output tensor of the layer).
        """
        ...
    
    def build(self, input_shape):
        """Creates the layer weights.

        Must be implemented on all layers that have weights.

        # Arguments
            input_shape: Keras tensor (future input to layer)
                or list/tuple of Keras tensors to reference
                for weight shape computations.
        """
        self.built = ...
    
    def _get_node_attribute_at_index(self, node_index, attr, attr_name):
        """Retrieves an attribute (e.g. input_tensors) from a node.

        This is used to implement the methods:
            - get_input_shape_at
            - get_output_shape_at
            - get_input_at
            etc...

        # Arguments
            node_index: Integer index of the node from which
                to retrieve the attribute.
            attr: Exact node attribute name.
            attr_name: Human-readable attribute name, for error messages.

        # Returns
            The layer's attribute `attr` at the node of index `node_index`.

        # Raises
            RuntimeError: If the layer has no inbound nodes.
            ValueError: If the index is does not match any node.
        """
        ...
    
    def get_input_shape_at(self, node_index):
        """Retrieves the input shape(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A shape tuple
            (or list of shape tuples if the layer has multiple inputs).
        """
        ...
    
    def get_output_shape_at(self, node_index):
        """Retrieves the output shape(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A shape tuple
            (or list of shape tuples if the layer has multiple outputs).
        """
        ...
    
    def get_input_at(self, node_index):
        """Retrieves the input tensor(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A tensor (or list of tensors if the layer has multiple inputs).
        """
        ...
    
    def get_output_at(self, node_index):
        """Retrieves the output tensor(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A tensor (or list of tensors if the layer has multiple outputs).
        """
        ...
    
    def get_input_mask_at(self, node_index):
        """Retrieves the input mask tensor(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A mask tensor
            (or list of tensors if the layer has multiple inputs).
        """
        ...
    
    def get_output_mask_at(self, node_index):
        """Retrieves the output mask tensor(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A mask tensor
            (or list of tensors if the layer has multiple outputs).
        """
        ...
    
    @property
    def input(self):
        """Retrieves the input tensor(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        # Returns
            Input tensor or list of input tensors.

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        """
        ...
    
    @property
    def output(self):
        """Retrieves the output tensor(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        # Returns
            Output tensor or list of output tensors.

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        """
        ...
    
    @property
    def input_mask(self):
        """Retrieves the input mask tensor(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        # Returns
            Input mask tensor (potentially None) or list of input
            mask tensors.

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        """
        ...
    
    @property
    def output_mask(self):
        """Retrieves the output mask tensor(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        # Returns
            Output mask tensor (potentially None) or list of output
            mask tensors.

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        """
        ...
    
    @property
    def input_shape(self):
        """Retrieves the input shape tuple(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        # Returns
            Input shape tuple
            (or list of input shape tuples, one tuple per input tensor).

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        """
        ...
    
    @property
    def output_shape(self):
        """Retrieves the output shape tuple(s) of a layer.

        Only applicable if the layer has one inbound node,
        or if all inbound nodes have the same output shape.

        # Returns
            Output shape tuple
            (or list of input shape tuples, one tuple per output tensor).

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        """
        ...
    
    @property
    def metrics(self):
        ...
    
    def add_metric(self, value, name: Optional[Any] = ...):
        """Adds metric tensor to the layer.

        # Arguments
            value: Metric tensor.
            name: String metric name.
        """
        ...
    
    def add_loss(self, losses, inputs: Optional[Any] = ...):
        """Adds losses to the layer.

        The loss may potentially be conditional on some inputs tensors,
        for instance activity losses are conditional on the layer's inputs.

        # Arguments
            losses: loss tensor or list of loss tensors
                to add to the layer.
            inputs: input tensor or list of inputs tensors to mark
                the losses as conditional on these inputs.
                If None is passed, the loss is assumed unconditional
                (e.g. L2 weight regularization, which only depends
                on the layer's weights variables, not on any inputs tensors).
        """
        ...
    
    def add_update(self, updates, inputs: Optional[Any] = ...):
        """Adds updates to the layer.

        The updates may potentially be conditional on some inputs tensors,
        for instance batch norm updates are conditional on the layer's inputs.

        # Arguments
            updates: update op or list of update ops
                to add to the layer.
            inputs: input tensor or list of inputs tensors to mark
                the updates as conditional on these inputs.
                If None is passed, the updates are assumed unconditional.
        """
        ...
    
    def get_updates_for(self, inputs):
        ...
    
    def get_losses_for(self, inputs):
        ...
    
    @property
    def weights(self):
        ...
    
    @K.eager
    def set_weights(self, weights):
        """Sets the weights of the layer, from Numpy arrays.

        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the layer (i.e. it should match the
                output of `get_weights`).

        # Raises
            ValueError: If the provided weights list does not match the
                layer's specifications.
        """
        ...
    
    @K.eager
    def get_weights(self):
        """Returns the current weights of the layer.

        # Returns
            Weights values as a list of numpy arrays.
        """
        ...
    
    def get_config(self):
        """Returns the config of the layer.

        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.

        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by `Network` (one layer of abstraction above).

        # Returns
            Python dictionary.
        """
        ...
    
    @classmethod
    def from_config(cls, config):
        """Creates a layer from its config.

        This method is the reverse of `get_config`,
        capable of instantiating the same layer from the config
        dictionary. It does not handle layer connectivity
        (handled by Network), nor weights (handled by `set_weights`).

        # Arguments
            config: A Python dictionary, typically the
                output of get_config.

        # Returns
            A layer instance.
        """
        ...
    
    def count_params(self):
        """Counts the total number of scalars composing the weights.

        # Returns
            An integer count.

        # Raises
            RuntimeError: if the layer isn't yet built
                (in which case its weights aren't yet defined).
        """
        ...
    
    def _get_existing_metric(self, name: Optional[Any] = ...):
        ...
    
    def __setattr__(self, name, value):
        ...
    


def _create_mean_metric(value, name: Optional[Any] = ...):
    ...

@K.symbolic
def _call_metric(metric_obj, *args, **kwargs):
    ...

class InputSpec(object):
    """Specifies the ndim, dtype and shape of every input to a layer.

    Every layer should expose (if appropriate) an `input_spec` attribute:
    a list of instances of InputSpec (one per input tensor).

    A None entry in a shape is compatible with any dimension,
    a None shape is compatible with any shape.

    # Arguments
        dtype: Expected datatype of the input.
        shape: Shape tuple, expected shape of the input
            (may include None for unchecked axes).
        ndim: Integer, expected rank of the input.
        max_ndim: Integer, maximum rank of the input.
        min_ndim: Integer, minimum rank of the input.
        axes: Dictionary mapping integer axes to
            a specific dimension value.
    """
    def __init__(self, dtype: Optional[Any] = ..., shape: Optional[Any] = ..., ndim: Optional[Any] = ..., max_ndim: Optional[Any] = ..., min_ndim: Optional[Any] = ..., axes: Optional[Any] = ...):
        self.dtype = ...
        self.shape = ...
        self.max_ndim = ...
        self.min_ndim = ...
        self.axes = ...
    
    def __repr__(self):
        ...
    


class Node(object):
    """A `Node` describes the connectivity between two layers.

    Each time a layer is connected to some new input,
    a node is added to `layer._inbound_nodes`.
    Each time the output of a layer is used by another layer,
    a node is added to `layer._outbound_nodes`.

    # Arguments
        outbound_layer: the layer that takes
            `input_tensors` and turns them into `output_tensors`
            (the node gets created when the `call`
            method of the layer was called).
        inbound_layers: a list of layers, the same length as `input_tensors`,
            the layers from where `input_tensors` originate.
        node_indices: a list of integers, the same length as `inbound_layers`.
            `node_indices[i]` is the origin node of `input_tensors[i]`
            (necessary since each inbound layer might have several nodes,
            e.g. if the layer is being shared with a different data stream).
        tensor_indices: a list of integers,
            the same length as `inbound_layers`.
            `tensor_indices[i]` is the index of `input_tensors[i]` within the
            output of the inbound layer
            (necessary since each inbound layer might
            have multiple tensor outputs, with each one being
            independently manipulable).
        input_tensors: list of input tensors.
        output_tensors: list of output tensors.
        input_masks: list of input masks (a mask can be a tensor, or None).
        output_masks: list of output masks (a mask can be a tensor, or None).
        input_shapes: list of input shape tuples.
        output_shapes: list of output shape tuples.
        arguments: dictionary of keyword arguments that were passed to the
            `call` method of the layer at the call that created the node.

    `node_indices` and `tensor_indices` are basically fine-grained coordinates
    describing the origin of the `input_tensors`, verifying the following:

    origin_node = inbound_layers[i]._inbound_nodes[node_indices[i]]
    input_tensors[i] == origin_node.output_tensors[tensor_indices[i]]

    A node from layer A to layer B is added to:
        A._outbound_nodes
        B._inbound_nodes
    """
    def __init__(self, outbound_layer, inbound_layers, node_indices, tensor_indices, input_tensors, output_tensors, input_masks, output_masks, input_shapes, output_shapes, arguments: Optional[Any] = ...):
        self.outbound_layer = ...
        self.inbound_layers = ...
        self.node_indices = ...
        self.tensor_indices = ...
        self.input_tensors = ...
        self.output_tensors = ...
        self.input_masks = ...
        self.output_masks = ...
        self.input_shapes = ...
        self.output_shapes = ...
        self.arguments = ...
    
    def get_config(self):
        ...
    


def _collect_previous_mask(input_tensors):
    """Retrieves the output mask(s) of the previous node.

    # Arguments
        input_tensors: A tensor or list of tensors.

    # Returns
        A mask tensor or list of mask tensors.
    """
    ...

def _to_snake_case(name):
    ...

def _collect_input_shape(input_tensors):
    """Collects the output shape(s) of a list of Keras tensors.

    # Arguments
        input_tensors: list of input tensors (or single input tensor).

    # Returns
        List of shape tuples (or single tuple), one tuple per input.
    """
    ...


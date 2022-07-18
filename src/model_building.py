from src.utils import metrics_define, WeightCalculator
import warnings
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Reshape, Permute,
                                     GlobalAveragePooling2D, GlobalMaxPooling2D,
                                     Input, MaxPooling2D, add, multiply)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.backend import is_keras_tensor
from tensorflow.keras.utils import get_source_inputs
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Input

from src.base import ModelBuilderBase

tfk = tf.keras
tfkl = tfk.layers


class AdjModel(ModelBuilderBase):

    @staticmethod
    def _tensor_shape(tensor):
        return getattr(tensor, 'shape')

    @staticmethod
    def _obtain_input_shape(input_shape,
                            default_size,
                            min_size,
                            data_format,
                            require_flatten,
                            weights=None):
        """Internal utility to compute/validate a model's tensor shape.
        # Arguments
            input_shape: Either None (will return the default network input shape),
                or a user-provided shape to be validated.
            default_size: Default input width/height for the model.
            min_size: Minimum input width/height accepted by the model.
            data_format: Image data format to use.
            require_flatten: Whether the model is expected to
                be linked to a classifier via a Flatten layer.
            weights: One of `None` (random initialization)
                or 'imagenet' (pre-training on ImageNet).
                If weights='imagenet' input channels must be equal to 3.
        # Returns
            An integer shape tuple (may include None entries).
        # Raises
            ValueError: In case of invalid argument values.
        """
        if weights != 'imagenet' and input_shape and len(input_shape) == 3:
            if data_format == 'channels_first':
                if input_shape[0] not in {1, 3}:
                    warnings.warn(
                        'This model usually expects 1 or 3 input channels. '
                        'However, it was passed an input_shape with {input_shape}'
                        ' input channels.'.format(input_shape=input_shape[0]))
                default_shape = (input_shape[0], default_size, default_size)
            else:
                if input_shape[-1] not in {1, 3}:
                    warnings.warn(
                        'This model usually expects 1 or 3 input channels. '
                        'However, it was passed an input_shape with {n_input_channels}'
                        ' input channels.'.format(n_input_channels=input_shape[-1]))
                default_shape = (default_size, default_size, input_shape[-1])
        else:
            if data_format == 'channels_first':
                default_shape = (3, default_size, default_size)
            else:
                default_shape = (default_size, default_size, 3)
        if weights == 'imagenet' and require_flatten:
            if input_shape is not None:
                if input_shape != default_shape:
                    raise ValueError('When setting `include_top=True` '
                                     'and loading `imagenet` weights, '
                                     '`input_shape` should be {default_shape}.'.format(default_shape=default_shape))
            return default_shape
        if input_shape:
            if data_format == 'channels_first':
                if input_shape is not None:
                    if len(input_shape) != 3:
                        raise ValueError(
                            '`input_shape` must be a tuple of three integers.')
                    if input_shape[0] != 3 and weights == 'imagenet':
                        raise ValueError('The input must have 3 channels; got '
                                         '`input_shape={input_shape}`'.format(input_shape=input_shape))
                    if ((input_shape[1] is not None and input_shape[1] < min_size) or
                            (input_shape[2] is not None and input_shape[2] < min_size)):
                        raise ValueError('Input size must be at least {min_size}x{min_size};'
                                         ' got `input_shape={input_shape}`'.format(min_size=min_size,
                                                                                   input_shape=input_shape))
            else:
                if input_shape is not None:
                    if len(input_shape) != 3:
                        raise ValueError(
                            '`input_shape` must be a tuple of three integers.')
                    if input_shape[-1] != 3 and weights == 'imagenet':
                        raise ValueError('The input must have 3 channels; got '
                                         '`input_shape={input_shape}`'.format(input_shape=input_shape))
                    if ((input_shape[0] is not None and input_shape[0] < min_size) or
                            (input_shape[1] is not None and input_shape[1] < min_size)):
                        raise ValueError('Input size must be at least {min_size}x{min_size};'
                                         ' got `input_shape={input_shape}`'.format(min_size=min_size,
                                                                                   input_shape=input_shape))
        else:
            if require_flatten:
                input_shape = default_shape
            else:
                if data_format == 'channels_first':
                    input_shape = (3, None, None)
                else:
                    input_shape = (None, None, 3)
        if require_flatten:
            if None in input_shape:
                raise ValueError('If `include_top` is True, '
                                 'you should specify a static `input_shape`. '
                                 'Got `input_shape={input_shape}`'.format(input_shape=input_shape))
        return input_shape

    def squeeze_excite_block(self, input_tensor, ratio=16):
        """ Create a channel-wise squeeze-excite block
        Args:
            input_tensor: input Keras tensor
            ratio: number of output filters
        Returns: a Keras tensor
        References
        -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
        """
        init = input_tensor
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = self._tensor_shape(init)[channel_axis]
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        if K.image_data_format() == 'channels_first':
            se = Permute((3, 1, 2))(se)

        x = multiply([init, se])
        return x

    def _resnet_block(self, input_tensor, filters, k=1, strides=(1, 1)):
        """ Adds a pre-activation resnet block without bottleneck layers
        Args:
            input_tensor: input Keras tensor
            filters: number of output filters
            k: width factor
            strides: strides of the convolution layer
        Returns: a Keras tensor
        """
        init = input_tensor
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1

        x = BatchNormalization(axis=channel_axis)(input_tensor)
        x = Activation('relu')(x)

        if strides != (1, 1) or self._tensor_shape(init)[channel_axis] != filters * k:
            init = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                          use_bias=False, strides=strides)(x)

        x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
                   use_bias=False, strides=strides)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
                   use_bias=False)(x)

        # squeeze and excite block
        x = self.squeeze_excite_block(x)

        m = add([x, init])
        return m

    def _resnet_bottleneck_block(self, input_tensor, filters, k=1, strides=(1, 1)):
        """ Adds a pre-activation resnet block with bottleneck layers
        Args:
            input_tensor: input Keras tensor
            filters: number of output filters
            k: width factor
            strides: strides of the convolution layer
        Returns: a Keras tensor
        """
        init = input_tensor
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        bottleneck_expand = 4

        x = BatchNormalization(axis=channel_axis)(input_tensor)
        x = Activation('relu')(x)

        if strides != (1, 1) or self._tensor_shape(init)[channel_axis] != bottleneck_expand * filters * k:
            init = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                          use_bias=False, strides=strides)(x)

        x = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                   use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
                   use_bias=False, strides=strides)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)

        x = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                   use_bias=False)(x)

        # squeeze and excite block
        x = self.squeeze_excite_block(x)

        m = add([x, init])
        return m

    def _create_se_resnet(self, classes, img_input, include_top, initial_conv_filters, filters,
                          depth, width, bottleneck, weight_decay, pooling):

        """Creates a SE ResNet model with specified parameters
        Args:
            initial_conv_filters: number of features for the initial convolution
            include_top: Flag to include the last dense layer
            filters: number of filters per block, defined as a list.
                filters = [64, 128, 256, 512]
            depth: number or layers in the each block, defined as a list.
                ResNet-50  = [3, 4, 6, 3]
                ResNet-101 = [3, 6, 23, 3]
                ResNet-152 = [3, 8, 36, 3]
            width: width multiplier for network (for Wide ResNet)
            bottleneck: adds a bottleneck conv to reduce computation
            weight_decay: weight_decay (l2 norm)
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
        Returns: a Keras Model
        """
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        N = list(depth)

        # block 1 (initial conv block)
        x = Conv2D(initial_conv_filters, (7, 7), padding='same', use_bias=False, strides=(2, 2),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # block 2 (projection block)
        for i in range(N[0]):
            if bottleneck:
                x = self._resnet_bottleneck_block(x, filters[0], width)
            else:
                x = self._resnet_block(x, filters[0], width)

        # block 3 - N
        for k in range(1, len(N)):
            if bottleneck:
                x = self._resnet_bottleneck_block(x, filters[k], width, strides=(2, 2))
            else:
                x = self._resnet_block(x, filters[k], width, strides=(2, 2))

            for i in range(N[k] - 1):
                if bottleneck:
                    x = self._resnet_bottleneck_block(x, filters[k], width)
                else:
                    x = self._resnet_block(x, filters[k], width)

        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)

        if include_top:
            x = GlobalAveragePooling2D()(x)
            x = Dense(classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                      activation='softmax', name='ah')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        return x

    def SEResNet(self, input_shape=None,
                 initial_conv_filters=64,
                 depth=[3, 4, 6, 3],
                 filters=[64, 128, 256, 512],
                 width=1,
                 bottleneck=False,
                 weight_decay=1e-4,
                 include_top=True,
                 weights=None,
                 input_tensor=None,
                 pooling=None,
                 classes=1000):

        """ Instantiate the Squeeze and Excite ResNet architecture. Note that ,
            when using TensorFlow for best performance you should set
            `image_data_format="channels_last"` in your Keras config
            at ~/.keras/keras.json.
            The model are compatible with both
            TensorFlow and Theano. The dimension ordering
            convention used by the model is the one
            specified in your Keras config file.
            # Arguments
                initial_conv_filters: number of features for the initial convolution
                depth: number or layers in the each block, defined as a list.
                    ResNet-50  = [3, 4, 6, 3]
                    ResNet-101 = [3, 6, 23, 3]
                    ResNet-152 = [3, 8, 36, 3]
                filter: number of filters per block, defined as a list.
                    filters = [64, 128, 256, 512
                width: width multiplier for the network (for Wide ResNets)
                bottleneck: adds a bottleneck conv to reduce computation
                weight_decay: weight decay (l2 norm)
                include_top: whether to include the fully-connected
                    layer at the top of the network.
                weights: `None` (random initialization) or `imagenet` (trained
                    on ImageNet)
                input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                    to use as image input for the model.
                input_shape: optional shape tuple, only to be specified
                    if `include_top` is False (otherwise the input shape
                    has to be `(224, 224, 3)` (with `tf` dim ordering)
                    or `(3, 224, 224)` (with `th` dim ordering).
                    It should have exactly 3 inputs channels,
                    and width and height should be no smaller than 8.
                    E.g. `(200, 200, 3)` would be one valid value.
                pooling: Optional pooling mode for feature extraction
                    when `include_top` is `False`.
                    - `None` means that the output of the model will be
                        the 4D tensor output of the
                        last convolutional layer.
                    - `avg` means that global average pooling
                        will be applied to the output of the
                        last convolutional layer, and thus
                        the output of the model will be a 2D tensor.
                    - `max` means that global max pooling will
                        be applied.
                classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
            # Returns
                A Keras model instance.
            """

        if weights not in {'imagenet', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `imagenet` '
                             '(pre-training on ImageNet).')

        if weights == 'imagenet' and include_top and classes != 1000:
            raise ValueError('If using `weights` as imagenet with `include_top`'
                             ' as true, `classes` should be 1000')

        assert len(depth) == len(filters), "The length of filter increment list must match the length " \
                                           "of the depth list."

        img_input = Input(input_shape)

        x = self._create_se_resnet(classes, img_input, include_top, initial_conv_filters,
                              filters, depth, width, bottleneck, weight_decay, pooling)

        # # Ensure that the model takes into account
        # # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input

        # print(x)
        # Create model.
        model = Model(inputs, x, name='se-resnet')

        # load weights

        return model

    def SEResNet50(self, input_shape=None,
                   width=1,
                   bottleneck=True,
                   weight_decay=1e-4,
                   include_top=True,
                   weights=None,
                   input_tensor=None,
                   pooling=None,
                   classes=1000):
        return self.SEResNet(input_shape,
                        width=width,
                        bottleneck=bottleneck,
                        weight_decay=weight_decay,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes)

    @staticmethod
    def model2():
        inputs = Input(shape=(3, 1, 1))
        x = Flatten()(inputs)
        x = Dense(units=128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(units=128, activation='relu')(x)
        x = BatchNormalization()(x)
        outputs = Dense(units=1, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def get_compiled_model(self):
        inputs = Input(shape=(3, 256, 256, 3))

        input1 = inputs[:, 0]
        input2 = inputs[:, 1]
        input3 = inputs[:, 2]

        featureExtractor = self.SEResNet50(input_shape=(256, 256, 3), include_top=True, classes=1)
        feature1 = featureExtractor(input1)
        feature2 = featureExtractor(input2)
        feature3 = featureExtractor(input3)

        output_feature = tf.stack((feature1, feature2, feature3), axis=1)

        refine = self.model2()
        overall_out = refine(output_feature)

        model = Model(inputs=inputs, outputs=overall_out)
        optimizer = Adam(learning_rate=self.config.model_builder.initial_learning_rate)
        metrics_all = metrics_define(len(self.config.class_names))
        weight_calculator = WeightCalculator(self.config)
        model.compile(optimizer=optimizer, loss=weight_calculator.get_weighted_loss(), metrics=metrics_all)

        return model



class DenseNet(ModelBuilderBase):

    def get_compiled_model(self) -> tfk.Model:
        mb_conf = self.config.model_builder
        input_shape = mb_conf.input_shape
        n_classes = 2

        input_tensor = tfkl.Input(input_shape, dtype=tf.uint8)  # encoded PNG inputs
        x = tf.cast(input_tensor, tf.float32)
        x = tfk.applications.densenet.preprocess_input(x)

        base_model = tfk.applications.densenet.DenseNet121(
            include_top=False,
            input_tensor=x,
            weights=mb_conf.weights,
            pooling="avg"
        )
        x = base_model(x)
        predictions = tfkl.Dense(n_classes,
                                 activation=mb_conf.activation,
                                 name="new_predictions")(x)
        model = tfk.Model(inputs=input_tensor, outputs=predictions)

        optimizer = tfk.optimizers.Adam(learning_rate=2.5e-6)
        metrics = [tfk.metrics.SensitivityAtSpecificity(0.8),
                   tfk.metrics.AUC(curve='PR', name='AUC of Precision-Recall Curve'),
                   tfk.metrics.FalseNegatives(),
                   tfk.metrics.FalsePositives(),
                   tfk.metrics.TrueNegatives(),
                   tfk.metrics.TruePositives()]

        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=metrics)

        return model


class ResNet(ModelBuilderBase):

    def get_compiled_model(self) -> tfk.Model:
        mb_conf = self.config.model_builder
        input_shape = mb_conf.input_shape
        n_classes = 2

        input_tensor = tfkl.Input(input_shape, dtype=tf.uint8)  # encoded PNG inputs
        x = tf.cast(input_tensor, tf.float32)
        x = tfk.applications.resnet.preprocess_input(x)

        base_model = tfk.applications.resnet.ResNet50(
            include_top=False,
            input_tensor=x,
            weights=mb_conf.weights,
            pooling="avg"
        )
        x = base_model(x)
        predictions = tfkl.Dense(n_classes - 1,
                                 activation=mb_conf.activation,
                                 name="new_predictions")(x)
        model = tfk.Model(inputs=input_tensor, outputs=predictions)

        optimizer = tfk.optimizers.Adam(learning_rate=2.5e-6)
        metrics = [tfk.metrics.SensitivityAtSpecificity(0.8),
                   tfk.metrics.AUC(curve='PR', name='AUC of Precision-Recall Curve'),
                   tfk.metrics.FalseNegatives(),
                   tfk.metrics.FalsePositives(),
                   tfk.metrics.TrueNegatives(),
                   tfk.metrics.TruePositives()]

        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=metrics)

        return model


class MobileNet(ModelBuilderBase):

    def get_compiled_model(self) -> tfk.Model:
        mb_conf = self.config.model_builder
        input_shape = mb_conf.input_shape
        n_classes = 2

        input_tensor = tfkl.Input(input_shape, dtype=tf.uint8)  # encoded PNG inputs
        x = tf.cast(input_tensor, tf.float32)
        x = tfk.applications.mobilenet.preprocess_input(x)

        base_model = tfk.applications.mobilenet.MobileNet(
            include_top=False,
            input_tensor=x,
            weights=mb_conf.weights,
            pooling="avg"
        )
        x = base_model(x)
        predictions = tfkl.Dense(n_classes - 1,
                                 activation=mb_conf.activation,
                                 name="new_predictions")(x)
        model = tfk.Model(inputs=input_tensor, outputs=predictions)

        optimizer = tfk.optimizers.Adam(learning_rate=2.5e-6)
        metrics = [tfk.metrics.SensitivityAtSpecificity(0.8),
                   tfk.metrics.AUC(curve='PR', name='AUC of Precision-Recall Curve'),
                   tfk.metrics.FalseNegatives(),
                   tfk.metrics.FalsePositives(),
                   tfk.metrics.TrueNegatives(),
                   tfk.metrics.TruePositives()]

        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=metrics)

        return model

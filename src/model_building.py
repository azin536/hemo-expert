import tensorflow as tf

from .base import ModelBuilderBase

tfk = tf.keras
tfkl = tfk.layers


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

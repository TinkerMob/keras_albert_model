import unittest
from .albert import keras
from .albert import get_custom_objects, build_albert


class TestALBERT(unittest.TestCase):

    def test_build_train(self):
        model = build_albert(333)
        model.compile('adam', 'sparse_categorical_crossentropy')
        model.save('train.h5')
        model = keras.models.load_model('train.h5',
                                        custom_objects=get_custom_objects())
        model.summary()

    def test_build_infer(self):
        model = keras.models.Model(*build_albert(345, training=False))
        model.compile('adam', 'sparse_categorical_crossentropy')
        model.save('infer.h5')
        model = keras.models.load_model('infer.h5',
                                        custom_objects=get_custom_objects())
        model.summary()

    def test_build_select_output_layer(self):
        model = build_albert(346, output_layers=-10, training=False)
        model.compile('adam', 'sparse_categorical_crossentropy')
        model.summary()

    def test_build_output_layers(self):
        model = build_albert(346,
                             output_layers=[-1, -2, -3, -4],
                             training=False)
        model.compile('adam', 'sparse_categorical_crossentropy')
        model.summary()

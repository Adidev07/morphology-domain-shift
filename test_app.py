import unittest
from app import *
class DomainTests(unittest.TestCase):
    def test_normalized_invariance(self):
        im=np.arange(144.).reshape(1,12,12);np.testing.assert_allclose(features(im,True),features(im*2+3,True))
    def test_finite_constant_image(self):self.assertTrue(np.isfinite(features(np.ones((2,12,12)),True)).all())
    def test_neural_learning(self):
        x=np.r_[np.zeros((40,3)),np.ones((40,3))];y=np.r_[np.zeros(40,int),np.ones(40,int)];self.assertGreater(np.mean(predict(fit(x,y,100),x)==y),.9)

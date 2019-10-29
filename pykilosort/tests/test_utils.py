import numpy as np
from numpy.testing import assert_array_equal as ae

from ..utils import Context


def test_context_1(tmp_path):
    arr = np.random.randn(10, 20)
    c = Context(tmp_path)
    c.intermediate.test1 = arr
    ae(c.read('test1'), arr)
    c.write(test1=arr)
    ae(c.read('test1'), arr)

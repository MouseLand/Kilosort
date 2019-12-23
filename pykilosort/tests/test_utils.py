import numpy as np
from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_allclose as ac
from pytest import raises

from ..utils import Context, LargeArrayWriter, memmap_large_array


def test_context_1(tmp_path):
    arr = np.random.randn(10, 20)

    c = Context(tmp_path)
    c.intermediate.test1 = arr

    ae(c.read('test1'), arr)

    c.write(test1=arr)
    ae(c.read('test1'), arr)

    c.write(test2=12.34)
    assert c.read('test2') == 12.34


def test_context_2(tmp_path):
    arr = np.random.randn(10, 20)

    c = Context(tmp_path)

    c.intermediate.test1 = arr
    c.intermediate.test2 = 12.34
    c.save()

    c.load()
    ae(c.intermediate.test1, arr)
    assert c.intermediate.test2 == 12.34

    c = Context(tmp_path)
    assert 'test1' not in c.intermediate
    c.load()
    ae(c.intermediate.test1, arr)
    assert c.intermediate.test2 == 12.34


def test_large_array_1(tmp_path):
    path = tmp_path / 'arr.dat'
    arr = np.random.rand(1000, 13).T

    law = LargeArrayWriter(path, dtype=arr.dtype, shape=(13, -1))
    with raises(AssertionError):
        law.append(np.random.rand(1000, 13))
    law.append(arr)
    law.close()

    arr_1 = memmap_large_array(path)
    assert arr_1.shape == arr.shape
    ac(arr_1, arr)


def test_large_array_2(tmp_path):
    path = tmp_path / 'arr.dat'
    arr = np.random.rand(1000, 13).T

    law = LargeArrayWriter(path, dtype=arr.dtype, shape=(13, -1))
    k = 10
    for i in range(arr.shape[-1] // k):
        law.append(arr[:, k * i: k * (i + 1)])
    law.close()

    arr_1 = memmap_large_array(path)
    assert arr_1.shape == arr.shape
    ac(arr_1, arr)


def test_large_array_3(tmp_path):
    path = tmp_path / 'arr.dat'

    law = LargeArrayWriter(path, dtype=np.float32, shape=(13, -1))
    arrs = []
    for i in range(100):
        arr = np.random.rand(np.random.randint(low=10, high=20), 13).T
        arrs.append(arr)
        law.append(arr)
    law.close()

    arr = np.concatenate(arrs, axis=1)
    arr_1 = memmap_large_array(path)
    assert arr_1.shape == arr.shape
    ac(arr_1, arr)

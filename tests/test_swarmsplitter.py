import numpy as np

from kilosort import swarmsplitter


def refractory_train(rng, rate=10.0, refractory=0.003, duration=1200.0):
    # A single unit firing at ~rate Hz with an absolute refractory period.
    isi = rng.exponential(1/rate, 20000) + refractory
    st = np.cumsum(isi)
    return st[st < duration]


def test_check_ccg_empty_inputs():
    # Empty trains must be caught before compute_CCG, which calls .max()
    # on both arrays.
    assert swarmsplitter.check_CCG(np.array([]), np.array([1.0])) == (False, False)
    assert swarmsplitter.check_CCG(np.array([1.0]), np.array([])) == (False, False)
    assert swarmsplitter.check_CCG(np.array([])) == (False, False)


def test_check_ccg_zero_duration():
    # Both trains a single, identical spike: T == 0, nothing to evaluate.
    assert swarmsplitter.check_CCG(np.array([1.0]), np.array([1.0])) == (False, False)


def test_refractoriness_blocks_refractory_split():
    # Two random halves of the same refractory spike train are mutually
    # refractory: check_CCG must report cross_refractory and
    # refractoriness must veto the split.
    rng = np.random.default_rng(0)
    st = refractory_train(rng)
    mask = rng.random(st.size) < 0.5
    st1, st2 = np.sort(st[mask]), np.sort(st[~mask])
    assert st1.size > 1000 and st2.size > 1000

    is_refractory, cross_refractory = swarmsplitter.check_CCG(st1, st2)
    assert cross_refractory
    assert swarmsplitter.refractoriness(st1, st2) == 1


def test_refractoriness_allows_independent_units():
    # Two independent units are not mutually refractory, so a split that
    # separates them must not be vetoed.
    rng = np.random.default_rng(1)
    st1 = refractory_train(rng)
    st2 = refractory_train(rng)

    is_refractory, cross_refractory = swarmsplitter.check_CCG(st1, st2)
    assert not cross_refractory
    assert swarmsplitter.refractoriness(st1, st2) == 0

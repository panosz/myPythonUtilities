import numpy as np
import numpy.testing as nt
import pytest
from panos_utilities import wrap

wrap_2pi_list = [(0, 0),
                 (np.pi, np.pi),
                 (2*np.pi, 0),
                 (8.345, 8.345 - 2*np.pi),
                 (-8.345, -8.345 + 4*np.pi),
                 ]


@pytest.mark.parametrize("test_input,expected", wrap_2pi_list)
def test_wrap_2pi_scalar(test_input, expected):
    result = wrap.wrap_2pi(test_input)
    nt.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


wrap_2pi_input_array = np.array([x for x, _ in wrap_2pi_list])
wrap_2pi_expected_result = np.array([y for _, y in wrap_2pi_list])


def test_wrap_2pi_array():
    result = wrap.wrap_2pi(wrap_2pi_input_array)
    nt.assert_allclose(result, wrap_2pi_expected_result,
                       rtol=1e-12, atol=1e-12)


wrap_minus_pi_pi_list = [(0, 0),
                         (np.pi, -np.pi),
                         (2*np.pi, 0),
                         (8.345, 8.345 - 2*np.pi),
                         #  (-8.345, -8.345 + 4*np.pi),
                         ]


@pytest.mark.parametrize("test_input,expected", wrap_minus_pi_pi_list)
def test_wrap_minus_pi_pi_scalar(test_input, expected):
    result = wrap.wrap_minus_pi_pi(test_input)
    nt.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


wrap_minus_pi_pi_input_array = np.array([x for x, _ in wrap_minus_pi_pi_list])
wrap_minus_pi_pi_expected_result = np.array([y for _,
                                             y in wrap_minus_pi_pi_list])


def test_wrap_minus_pi_pi_array():
    result = wrap.wrap_minus_pi_pi(wrap_minus_pi_pi_input_array)
    nt.assert_allclose(result, wrap_minus_pi_pi_expected_result,
                       rtol=1e-12, atol=1e-12)

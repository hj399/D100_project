import numpy as np
import pytest

from d100_d400_project.modules.feature_engineering import CustomOrdinalEncoder


# Test for known values
@pytest.mark.parametrize(
    "data, input_data, expected_output",
    [
        (
            np.array([["A", "B"], ["B", "C"], ["A", "C"]]),  # Training data
            np.array([["A", "C"], ["B", "B"]]),  # Test data
            np.array([[0, 1], [1, 0]]),  # Expected encoded result
        ),
        (
            np.array([["X", "Y"], ["Y", "Z"], ["X", "Z"]]),  # Training data
            np.array([["Y", "Z"], ["X", "Y"]]),  # Test data
            np.array([[1, 1], [0, 0]]),  # Expected encoded result
        ),
    ],
)
def test_custom_ordinal_encoder_known_values(data, input_data, expected_output):
    """
    Test CustomOrdinalEncoder for expected behavior on known values.

    Parameters
    ----------
    data : numpy.ndarray
        Training data for fitting the encoder.
    input_data : numpy.ndarray
        Test data to be transformed using the fitted encoder.
    expected_output : numpy.ndarray
        Expected ordinal encoding output for the test data.

    Assertions
    ----------
    - Ensure that the transformed data matches the expected output.
    """
    encoder = CustomOrdinalEncoder()
    encoder.fit(data)
    transformed_data = encoder.transform(input_data)

    # Assert that the transformed data matches the expected output
    np.testing.assert_array_equal(
        transformed_data,
        expected_output,
        "Transformed data does not match expected output.",
    )


# Test for unknown values
@pytest.mark.parametrize(
    "data, input_data",
    [
        (
            np.array([["A", "B"], ["B", "C"]]),  # Training data
            np.array(
                [["A", "Unknown"], ["Unknown", "B"]]
            ),  # Test data with unknown values
        ),
    ],
)
def test_custom_ordinal_encoder_unknown_values(data, input_data):
    """
    Test CustomOrdinalEncoder for handling unknown values.

    Parameters
    ----------
    data : numpy.ndarray
        Training data for fitting the encoder.
    input_data : numpy.ndarray
        Test data with unknown values to be transformed.

    Assertions
    ----------
    - Ensure that unknown categories are correctly mapped to -1.
    """
    encoder = CustomOrdinalEncoder()
    encoder.fit(data)
    transformed_data = encoder.transform(input_data)

    assert (
        -1 in transformed_data.values
    ), "Unknown categories were not mapped to -1 as expected."

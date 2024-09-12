import pytest

def test_addition():
    """
    Test that the addition of two numbers is correct.
    """
    result = 1 + 1
    assert result == 2, "Addition result should be 2"

def test_string_uppercase():
    """
    Test that the upper() method correctly converts a string to uppercase.
    """
    sample_str = "hello"
    result = sample_str.upper()
    assert result == "HELLO", "String should be converted to uppercase"

def test_list_length():
    """
    Test that the length of a list is correctly calculated.
    """
    sample_list = [1, 2, 3]
    result = len(sample_list)
    assert result == 3, "List length should be 3"

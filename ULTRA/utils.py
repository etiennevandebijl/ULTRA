import numpy as np

def check_array_is_set(array_set):
    # Check if all elements are integers
    if not all(isinstance(x, np.int64) for x in array_set):
        raise ValueError("There is a non-integer value in the set")
        
    # Check if all elements are unique
    if len(array_set) != len(set(array_set)):
        raise ValueError("There is at least one duplicate value in the set")
        

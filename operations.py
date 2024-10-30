import numpy as np

def replace_nonfinite_in_place(x):
    x[np.logical_not(np.isfinite(x))] = 0
    
def replace_nans_out_of_place(x):
    return np.where(np.isnan(x), 0, x)

def find_mode(x):
    x = x.reshape(-1).astype(int)  # ensure x is a flattened int array
    a = np.bincount(x)
    return np.argmax(a)

def flip_and_slice_matrix(x):
    temp = x[:, ::-1]
    return temp[::3]

def divide_matrix_along_rows(x, y):
    z = y[:, np.newaxis]
    return x/z

    

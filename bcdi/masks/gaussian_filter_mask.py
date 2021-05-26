from scipy.ndimage import gaussian_filter
import numpy as np

filename = input("filename = ")
sigma = int(input("sigma = "))
threshold = int(input("threshold = "))

def gaussian_convolution(filename, sigma, threshold):
    """Apply a gaussian convolution to the mask, to avoid having holes inside"""
    
    old_mask = np.load(filename)["mask"]

    bigdata = 100 * old_mask
    conv_mask = np.where(gaussian_filter(bigdata, sigma) > threshold, 1, 0)
    
    np.savez(f"filter_sig{sigma}_t{threshold}_{filename}", oldmask = old_mask, mask = conv_mask)
    
    print(f"New mask saved as \nfilter_sig{sigma}_t{threshold}_{filename}")

gaussian_convolution(filename, sigma, threshold)
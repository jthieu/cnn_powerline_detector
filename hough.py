import cv2 as cv
import numpy as np

def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    ys, xs = np.nonzero(img)
    for i in range(0,len(xs)):
        y = ys[i]
        x = xs[i]
        for theta in range(0, len(thetas)):
            rho = x*cos_t[theta] + y*sin_t[theta]
            rho_index = round(rho) + diag_len
            accumulator[int(rho_index), int(theta)]+=1
   
    ### END YOUR CODE
    

    return accumulator, rhos, thetas

def main():
	img = cv2.imread('TV_VL_1753.bmg')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
	edges = cv2.Canny(gray,50,150,apertureSize = 3) 
	lines = hough_transform(img)
	print(img)
	print(lines) 
import numpy as np
import pylab
import cv2
pylab.ion()


def kernal_convolution(image_path):

    '''

    *Performs a Sobel Kernal Conolution and Saves Output as JPG.

	Parameters
	----------

	image_path : <type 'str'>
		Amplitude of the model light curve with units of signal to noise.

	Returns
	-------
	y_copy :  <type 'ndarray'>

    '''

    #Sobel kernal
    kernal_x = [[-1,0,1],
               [-2,0,2],
               [-1,0,1]]

    kernal_y = [[-1,-2,-1],
               [0,0,0],
               [1,2,1]]


    #open image and make copy
    img = cv2.imread(image_path)
    y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    xsize,ysize = y.shape
    y_copy = np.zeros((xsize,ysize))

    #loop through and calculate
    for i in range(1, xsize-1):

        for j in range(1, ysize - 1):


            xpix = (kernal_x[0][0] * y[i-1][j-1]) + (kernal_x[0][1] * y[i][j-1]) + (kernal_x[0][2] * y[i+1][j-1]) +\
                (kernal_x[1][0] * y[i-1][j])   + (kernal_x[1][1] * y[i][j])   + (kernal_x[1][2] * y[i+1][j]) +\
                (kernal_x[2][0] * y[i-1][j+1]) + (kernal_x[2][1] * y[i][j+1]) + (kernal_x[2][2] * y[i+1][j+1])

            ypix = (kernal_y[0][0] * y[i-1][j-1]) + (kernal_y[0][1] * y[i][j-1]) + (kernal_y[0][2] * y[i+1][j-1]) +\
                (kernal_y[1][0] * y[i-1][j])   + (kernal_y[1][1] * y[i][j])   + (kernal_y[1][2] * y[i+1][j]) +\
                (kernal_y[2][0] * y[i-1][j+1]) + (kernal_y[2][1] * y[i][j+1]) + (kernal_y[2][2] * y[i+1][j+1])


            mag = np.sqrt( (xpix * xpix) + (ypix * ypix) )
            mag = np.abs(mag)

            y_copy[i][j] = mag

    #normalise and save
    y_copy *= 255.0 / np.max(y_copy)
    pylab.imshow(y_copy, cmap = 'gray')
    pylab.savefig('output_' + image_path + '.jpg', dpi=200)

    return y_copy

if __name__ == '__main__':
    
    arr = kernal_convolution('corpus.jpg')

import matlab
import matlab.engine
import cv2
from imageio import imsave, imread
import numpy as np
import scipy.io as io

eng = matlab.engine.start_matlab()
# t = eng.myls(4,2)
img = imread('./test_images/00582.png')
img_mat = matlab.double(img.tolist())
m = eng.IDE(img_mat, 0.5)
print(m)
f_J_IDE = io.loadmat("ide_out/J_IDE.mat")
J = f_J_IDE["J"]
J = np.array(J)
cv2.imshow('J',J[:,:,::-1]);
f_t_IDE = io.loadmat("ide_out/t_refine.mat")
t = f_t_IDE["t_refine"]
t = np.array(t)
cv2.imshow('t', t);
cv2.waitKey();
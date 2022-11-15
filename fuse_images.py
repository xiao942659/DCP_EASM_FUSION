import numpy as np
import cv2

def fuse_images(real_I, rec_J, refine_J, t_DCP, t_IDE):
    """
    real_I, rec_J, and refine_J: Images with shape hxwx3
    """
    # realness features
    mat_RGB2YMN = np.array([[0.299,0.587,0.114],
                            [0.30,0.04,-0.35],
                            [0.34,-0.6,0.17]])

    recH,recW,recChl = rec_J.shape
    rec_J_flat = rec_J.reshape([recH*recW,recChl])
    rec_J_flat_YMN = (mat_RGB2YMN.dot(rec_J_flat.T)).T
    rec_J_YMN = rec_J_flat_YMN.reshape(rec_J.shape)

    refine_J_flat = refine_J.reshape([recH*recW,recChl])
    refine_J_flat_YMN = (mat_RGB2YMN.dot(refine_J_flat.T)).T
    refine_J_YMN = refine_J_flat_YMN.reshape(refine_J.shape)

    real_I_flat = real_I.reshape([recH*recW,recChl])
    real_I_flat_YMN = (mat_RGB2YMN.dot(real_I_flat.T)).T
    real_I_YMN = real_I_flat_YMN.reshape(real_I.shape)

    # gradient features
    rec_Gx = cv2.Sobel(rec_J_YMN[:,:,0],cv2.CV_64F,1,0,ksize=3)
    rec_Gy = cv2.Sobel(rec_J_YMN[:,:,0],cv2.CV_64F,0,1,ksize=3)
    rec_GM = np.sqrt(rec_Gx**2 + rec_Gy**2)

    refine_Gx = cv2.Sobel(refine_J_YMN[:,:,0],cv2.CV_64F,1,0,ksize=3)
    refine_Gy = cv2.Sobel(refine_J_YMN[:,:,0],cv2.CV_64F,0,1,ksize=3)
    refine_GM = np.sqrt(refine_Gx**2 + refine_Gy**2)

    real_Gx = cv2.Sobel(real_I_YMN[:,:,0],cv2.CV_64F,1,0,ksize=3)
    real_Gy = cv2.Sobel(real_I_YMN[:,:,0],cv2.CV_64F,0,1,ksize=3)
    real_GM = np.sqrt(real_Gx**2 + real_Gy**2)

    # similarity
    rec_S_V = (2*real_GM*rec_GM+160)/(real_GM**2+rec_GM**2+160)
    rec_S_M = (2*rec_J_YMN[:,:,1]*real_I_YMN[:,:,1]+130)/(rec_J_YMN[:,:,1]**2+real_I_YMN[:,:,1]**2+130)
    rec_S_N = (2*rec_J_YMN[:,:,2]*real_I_YMN[:,:,2]+130)/(rec_J_YMN[:,:,2]**2+real_I_YMN[:,:,2]**2+130)
    rec_S_R = (rec_S_M*rec_S_N).reshape([recH,recW])

    refine_S_V = (2*real_GM*refine_GM+160)/(real_GM**2+refine_GM**2+160)
    refine_S_M = (2*refine_J_YMN[:,:,1]*real_I_YMN[:,:,1]+130)/(refine_J_YMN[:,:,1]**2+real_I_YMN[:,:,1]**2+130)
    refine_S_N = (2*refine_J_YMN[:,:,2]*real_I_YMN[:,:,2]+130)/(refine_J_YMN[:,:,2]**2+real_I_YMN[:,:,2]**2+130)
    refine_S_R = (refine_S_M*refine_S_N).reshape([recH,recW])


    rec_S = rec_S_R*np.power(rec_S_V, 0.4)
    refine_S = refine_S_R*np.power(refine_S_V, 0.4)


    fuseWeight = np.exp(rec_S)/(np.exp(rec_S)+np.exp(refine_S))
    fuseWeightMap = fuseWeight.reshape([recH,recW,1]).repeat(3,axis=2)
    t_DCP3 = np.zeros_like(real_I)
    t_DCP3[:, :, 0] = t_DCP
    t_DCP3[:, :, 1] = t_DCP
    t_DCP3[:, :, 2] = t_DCP
    t_IDE3 = np.zeros_like(real_I)
    t_IDE3[:, :, 0] = t_IDE
    t_IDE3[:, :, 1] = t_IDE
    t_IDE3[:, :, 2] = t_IDE
    fuse_J = t_DCP3*fuseWeightMap + t_IDE3*(1-fuseWeightMap)
    return fuse_J
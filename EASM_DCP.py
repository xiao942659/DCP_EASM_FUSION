import glob
import math;

import cv2;
import numpy as np;
import scipy.io as io
from fuse_images import fuse_images
import matlab
import matlab.engine
from watershed import segmen_sky
from scipy.signal import convolve2d
import pandas as pd
my_A = []
name_list = []

def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz, 3);

    indices = darkvec.argsort();
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    # my_A.append(np.mean(A))
    return A

def segsky_AtmLight(image, dark):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    seg_list, labels = segmen_sky(image_gray)
    segimg = None
    posimg = None
    # 如果存在天空区域，使用分水岭分割天空区域，寻找天空区域内的最亮天空区域
    if len(seg_list):
        skyimg = np.zeros(image_gray.shape, image_gray.dtype)
        for each in seg_list:
            segimg = labels == each
            skyimg_temp = image_gray * segimg
            skyimg += skyimg_temp
            # cv2.imshow("skyimg", skyimg)
            # cv2.waitKey()
        skyimg = cv2.medianBlur(skyimg, 15)
        # cv2.imshow("skyimg", skyimg)
        # cv2.waitKey()
        kernel_size = int(np.sqrt(image_gray.size * 0.001))
        # 定义卷积核
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        print('skyimg[0]', skyimg.shape[0])
        candidate_A = convolve2d(skyimg, kernel, mode='valid')
        pos = np.unravel_index(np.argmax(candidate_A), candidate_A.shape)
        print(pos)
        posimg = image.copy()
        for i in range(pos[0], pos[0]+kernel_size):
            for j in range(pos[1], pos[1]+kernel_size):
                posimg[i, j] = [0, 0, 255]
        A = np.zeros([1, 3])
        for i in range(3):
            a = np.mean(image[pos[0]:pos[0] + kernel_size, pos[1]:pos[1] + kernel_size, 0]) / 255
            A[0, i] = a

        # A = np.zeros([1, 3])
        # a_num = np.count_nonzero(skyimg)
        # a = np.sum(skyimg)/a_num / 255
        # for i in range(3):
        #     A[0, i] = a
        # I = src.astype('float64') / 255
        # A = AtmLight(I, skyimg)

    else:
        # 如果不存在天空区域，使用DCP方法求大气光
        # 张楠是不是大气光求的不对，导致透射率为全白？ 我没有提前将图像 / 255 导致透射率全白
        I = src.astype('float64') / 255
        A = AtmLight(I, dark)

    my_A.append(np.mean(A))
    return A, segimg, posimg


def TransmissionEstimate(im, A, sz):
    omega = 0.95;
    im3 = np.empty(im.shape, im.dtype);

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz);
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r));
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r));
    cov_Ip = mean_Ip - mean_I * mean_p;

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r));
    var_I = mean_II - mean_I * mean_I;

    a = cov_Ip / (var_I + eps);
    b = mean_p - a * mean_I;

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r));
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r));

    q = mean_a * im + mean_b;
    return q;


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t


def Recover(im, t, A, tx=0.1):
    res_DCP = np.empty(im.shape, im.dtype);
    t = cv2.max(t, tx);
    lnt = np.log(t)
    lntmin = np.log(np.min(t))
    a = lnt / lntmin

    for ind in range(0, 3):
        res_DCP[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res_DCP

def Gaussian_pyramid(original_image, down_times):
    temp = original_image.copy()
    gaussian_pyramid = [temp]
    for i in range(down_times):
        temp = cv2.pyrDown(temp)
        gaussian_pyramid.append(temp)
    return gaussian_pyramid

def laplacian(gaussian_pyramid, up_times):
    laplacian_pyramid = [gaussian_pyramid[-1]]
    for i in range(up_times, 0, -1):
        # print(i)
        temp_pyrUp = cv2.pyrUp(gaussian_pyramid[i])
        h, w = gaussian_pyramid[i-1].shape[:2]
        temp_pyrUp = cv2.resize(temp_pyrUp, (w, h))
        temp_lap = cv2.subtract(gaussian_pyramid[i-1], temp_pyrUp)
        laplacian_pyramid.append(temp_lap)
    return laplacian_pyramid

def EASM(J_DCP, A, I, t_DCP, src):
    p = np.empty(J_DCP.shape, J_DCP.dtype);
    J = np.empty(J_DCP.shape, J_DCP.dtype);
    for ind in range(0, 3):
        p[:, :, ind] = J_DCP[:, :, ind] / A[0, ind]
    Graymean = float(np.mean(p))
    eng = matlab.engine.start_matlab()
    I_mat = matlab.double(I.tolist())
    res = eng.IDE(I_mat, Graymean)
    print(res)
    f_J_IDE = io.loadmat("ide_out/J_IDE.mat")
    J_IDE = f_J_IDE["J"]
    J_IDE = np.array(J_IDE)
    f_t_IDE = io.loadmat("ide_out/t_refine.mat")
    t_IDE = f_t_IDE["t_refine"]
    t_IDE = np.array(t_IDE)

    t_DCP_Gp = Gaussian_pyramid(t_DCP, 4)
    t_DCP_lap = laplacian(t_DCP_Gp, 4)

    t_IDE_Gp = Gaussian_pyramid(t_IDE, 4)
    t_IDE_lap = laplacian(t_IDE_Gp, 4)

    I_Gp = Gaussian_pyramid(I, 4)
    I_lap = laplacian(I_Gp, 4)

    J_DCP_Gp = Gaussian_pyramid(J_DCP, 4)
    J_DCP_lap = laplacian(J_DCP_Gp, 4)

    J_IDE_Gp = Gaussian_pyramid(J_IDE, 4)
    J_IDE_lap = laplacian(J_IDE_Gp, 4)

    t_fuse_lap = []
    for i in range(len(J_IDE_lap)):
        t_fuse = fuse_images(I_lap[i], J_DCP_lap[i], J_IDE_lap[i], t_DCP_lap[i], t_IDE_lap[i])
        t_fuse_lap.append(t_fuse)

    t_restruction = t_fuse_lap[0]
    for i in range(1, len(t_fuse_lap)):
        t_pyrUp = cv2.pyrUp(t_restruction)
        h, w = t_fuse_lap[i].shape[:2]
        t_pyrUp = cv2.resize(t_pyrUp, (w, h))
        t_restruction = t_pyrUp + t_fuse_lap[i]

    # t_fuse = fuse_images(I, J_DCP, J_IDE, t_DCP, t_IDE)
    t_restruction_refine = np.empty(J_DCP.shape, J_DCP.dtype);
    for ind in range(0, 3):
        t_restruction_refine[:, :, ind] = TransmissionRefine(src, t_restruction[:, :, ind])
    t_restruction_refine = np.clip(t_restruction_refine, 0.1, 1)
    for ind in range(0, 3):
        J[:, :, ind] = (I[:, :, ind] - A[0, ind]) / t_restruction_refine[:, :, ind] + A[0, ind]
    # cv2.imshow('t_restruction', t_restruction)
    # cv2.imshow('t_restruction_refine', t_restruction_refine)
    # cv2.imshow('t_IDE', t_IDE)
    # cv2.imshow('t_DCP', t_DCP)
    # cv2.imshow('J_IDE', J_DCP)
    # cv2.imshow('J_DCP', J_DCP)
    # cv2.imshow('I', I)
    # cv2.imshow('J', J)
    # cv2.waitKey()
    return J, t_restruction_refine, t_IDE


if __name__ == '__main__':
    import sys

    try:
        fn = sys.argv[1]
    except:
        fn = './image/00001.png'


    def nothing(*argv):
        pass


    hazy_list = glob.glob("test_A2/hazy/*")
    for img in hazy_list:
        src = cv2.imread(img);

        I = src.astype('float64') / 255;

        dark = DarkChannel(I, 15);
        # A = AtmLight(I, dark);
        A, segimg, posimg = segsky_AtmLight(src, dark)
        te = TransmissionEstimate(I, A, 15);
        t = TransmissionRefine(src, te);
        res_DCP = Recover(I, t, A, 0.1);
        J, t_restruction_refine, t_IDE = EASM(res_DCP, A, I, t, src)

        # cv2.imshow("dark",dark);
        # cv2.imshow("t",t);
        # cv2.imshow('I',src);
        # cv2.imshow('res_DCP', res_DCP);
        # cv2.imshow('res_EASM_DCP', res_EASM_DCP);
        # if not segimg is None:
        #     cv2.imwrite("./SOTS_seg_sky_img/" + img.split("\\")[-1], segimg.astype(int)*255)
        # if not posimg is None:
        #     cv2.imwrite("./SOTS_A_watershed/" + img.split("\\")[-1], posimg)
        # cv2.imwrite("./SOTS_test_result_easm_watershed/" + img.split("\\")[-1], J*255);
        cv2.imwrite("./test_A2/my/" + img.split("\\")[-1], J * 255);
        name_list.append(img.split("\\")[-1])
        # cv2.imwrite("./SOTS_t_IDE/" + img.split("\\")[-1], t_IDE * 255);
        # cv2.imwrite("./SOTS_t_DCP/" + img.split("\\")[-1], t * 255);

        # cv2.waitKey();
    dit = {'image_number': name_list, 'A': my_A}
    df = pd.DataFrame(dit)
    df.to_csv('./test_A2/my/my_out_A.csv',
              columns=['image_number', 'A'], index=False, sep=',')

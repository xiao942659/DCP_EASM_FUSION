import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology,color,data,filters
import cv2
from scipy.signal import convolve2d
import heapq

# image =color.rgb2gray(data.camera())
# denoised = filters.rank.median(image, morphology.disk(2)) #过滤噪声
def segmen_sky(image):
    GaussianBlur = cv2.GaussianBlur(image, (15,15), 0)

    #将梯度值低于10的作为开始标记点
    markers = filters.rank.gradient(GaussianBlur, morphology.disk(5)) < 10
    markers = ndi.label(markers)[0]

    gradient = filters.rank.gradient(GaussianBlur, morphology.disk(2)) #计算梯度
    labels =morphology.watershed(gradient, markers, mask=image) #基于梯度的分水岭算法


    seg_list = []
    segimg_mean_list = []
    segimg_num_list = []
    for i in range(1, np.max(labels)+1):
        num = np.sum(labels == i)
        segimg_num_list.append(num)
        print('num', num)
        lab_sum = np.sum(image * (labels == i))
        print('lab_sum', lab_sum)
        segimg_mean = lab_sum / num
        segimg_mean_list.append(segimg_mean)
        print('segimg_mean', segimg_mean)
        if segimg_mean >= 205 and (num/labels.size >= 0.05):
            seg_list.append(i)



    # if len(seg_list):
    #     skyimg = np.empty(image.shape, image.dtype)
    #     for each in seg_list:
    #         segimg = labels == each
    #         skyimg_temp = image * segimg
    #         skyimg += skyimg_temp
    #         cv2.imshow("skyimg", skyimg)
    #         cv2.waitKey()
    #     skyimg = cv2.medianBlur(skyimg, 15)
    #     cv2.imshow("skyimg", skyimg)
    #     cv2.waitKey()
    # else:
    #     skyimg = None

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    axes = axes.ravel()
    ax0, ax1, ax2, ax3 = axes

    ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax0.set_title("(b) Gray", y=-0.15, fontsize=18)
    ax1.imshow(gradient, cmap=plt.cm.gray, interpolation='nearest')
    ax1.set_title("(c) Gradient", y=-0.15, fontsize=18)
    ax2.imshow(markers, cmap=plt.cm.gray, interpolation='nearest')
    ax2.set_title("(d) Markers",  y=-0.15, fontsize=18)
    ax3.imshow(labels==seg_list[0], cmap=plt.cm.gray, interpolation='nearest')
    ax3.set_title("(e) Segmented",  y=-0.15, fontsize=18)

    for ax in axes:
        ax.axis('off')

    fig.tight_layout()
    plt.show()

    return seg_list, labels

    # if len(seg_list) == 0:
    #     # re1 = list(map(segimg_mean_list.index, heapq.nlargest(3, segimg_mean_list)))  # 求最大的三个索引    nsmallest与nlargest相反，求最小
    #     # re2 = heapq.nlargest(3, segimg_mean_list)  # 求最大的三个元素
    #     # print(re1)  # 因为re1由map()生成的不是list，直接print不出来，添加list()就行了
    #     # print(re2)
    #     # for each in re1:
    #     #     num = np.sum(labels == each)
    #     #     if (num / labels.size >= 0.05):
    #     #         seg_list.append(each)
    #     for mean, num in zip(segimg_mean_list, segimg_num_list):
    #         if mean >= np.mean(image) and (num/labels.size >= 0.05) :
    #             seg_list.append(each)





def BrightChannel(im,sz):
    b,g,r = cv2.split(im)
    bc = cv2.max(cv2.max(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    bright = cv2.erode(bc,kernel)
    return bright

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    transmission = np.empty(im.shape,im.dtype);

    # transmission = 1 - omega*BrightChannel(im3,sz);
    for ind in range(0, 3):
        transmission[:,:,ind] = (im[:,:,ind]-0.8) / (omega * BrightChannel(im, sz) -0.8)
    return transmission

if __name__ == '__main__':
    image = cv2.imread('./HR/00502.png')
    image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    seg_list, labels = segmen_sky(image_gray)
    if len(seg_list):
        skyimg = np.zeros(image_gray.shape, image_gray.dtype)
        for each in seg_list:
            segimg = labels == each
            skyimg_temp = image_gray * segimg
            skyimg += skyimg_temp
            cv2.imshow("skyimg", skyimg)
            cv2.waitKey()
        skyimg = cv2.medianBlur(skyimg, 15)
        cv2.imshow("skyimg", skyimg)
        cv2.waitKey()
        kernel_size = int(np.sqrt(image_gray.size * 0.001))
        # 定义卷积核
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size*kernel_size)
        print('skyimg[0]',skyimg.shape[0])
        # candidate_A = np.zeros((skyimg.shape[0]-kernel_size, skyimg.shape[1]-kernel_size), skyimg.dtype)
        # candidate_A = cv2.filter2D(skyimg, -1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
        candidate_A = convolve2d(skyimg, kernel, mode='valid')
        hot = candidate_A
        candidate_A_hot = cv2.applyColorMap(hot.astype(np.uint8), cv2.COLORMAP_JET )  # 热图
        cv2.imwrite('./test_result/00502_hot_candidate_A.png', skyimg)
        # cv2.imshow("candidate_A_hot", candidate_A_hot)
        # cv2.waitKey()
        # plt.imshow(candidate_A_hot)
        # plt.colorbar()
        # plt.show()

        pos = np.unravel_index(np.argmax(candidate_A), candidate_A.shape)
        print(pos)
        A = np.zeros([1, 3])
        for i in range(3):
            a = np.mean(image[pos[0]:pos[0]+kernel_size, pos[1]:pos[1]+kernel_size, 0])
            A[0, i] = a

        posimg = image.copy()
        for i in range(pos[0], pos[0] + kernel_size):
            for j in range(pos[1], pos[1] + kernel_size):
                posimg[i, j] = [0, 0, 255]

        cv2.imshow("posimg", posimg)
        cv2.waitKey()
        # bright = BrightChannel(skyimg, 15)
        # I = image.astype('float64') / 255;
        # te = TransmissionEstimate(I, 0.8, 15);
        # cv2.imshow("te", te)
        # cv2.waitKey()
    else:
        skyimg = None

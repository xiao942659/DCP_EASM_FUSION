from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import cv2
import os
import pandas as pd
import numpy as np
import re

test_dir = r'./test_A2/my/'
GT_dir = r'./test_A2/clear/'
Hz_dir = './SOTS/HR_hazy/he_gt/'
test_files_name=[]
GT_files_name=[]

image_number = []
psnr_number = []
ssim_number = []

for root,dirs,files in os.walk(test_dir, topdown=False):
    test_files_name.append(files)
for i in test_files_name[0]:
    if not i.endswith('original.jpg'):
        temp = i.split('_')[0]+'.jpg'
        GT = cv2.imread(GT_dir + '%s' % (temp))
        print(i,'GT', GT.shape)
        test = cv2.imread(test_dir+'%s'% (i))
        GT = np.resize(GT, test.shape)
        print(i,'test', test.shape)

        # cv2.imwrite('./temp/'+'%s' % (i), test)
        # i = re.findall(r"\d+\.?\d*", str(i))
        # GT = cv2.imread(GT_dir + '%sjpg' % (i[0]))
        # h, w, c = GT.shape
        # test = cv2.resize(test, (w, h))

        # Hz = cv2.imread(Hz_dir+'%s'% (i))
        # cv2.imwrite('./SOTS/he_gt/%s'%(i),GT)
        # cv2.imwrite('./SOTS/he_hazy/%s' % (i), Hz)

        ssim = compare_ssim(GT, test, multichannel=True)
        psnr = compare_psnr(GT, test)

        image_number.append(i)
        psnr_number.append(psnr)
        ssim_number.append(ssim)

dit = {'image_number':image_number, 'psnr':psnr_number,'ssim':ssim_number}
df = pd.DataFrame(dit)
df.to_csv(r'./test_A2/my/my_metrics_result.csv',columns=['image_number','psnr','ssim'],index=False,sep=',')
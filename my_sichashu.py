import numpy as np
import cv2

#正方形类
class square:
    #lx、ly为左上角坐标(x对应行号，y对应列号)，rx、ry为右下角坐标
    def __init__(self,lx,ly,rx,ry):
        self.lx=lx
        self.ly=ly
        self.rx=rx
        self.ry=ry

    #打印正方形的坐标
    def print(self):
        print("lx:",self.lx,end=", ")
        print("ly:",self.ly,end=", ")
        print("rx:",self.rx,end=", ")
        print("ry:",self.ry)

def fenge(index_s, tu,square_arr):
    mat_temp = []
    mat_temp1 = []
    divide_block = []
    mean = []
    lx = square_arr[index_s].lx
    ly = square_arr[index_s].ly
    rx = square_arr[index_s].rx
    ry = square_arr[index_s].ry
    # 左上角
    divide_block0 = square(lx, ly, (lx + rx) // 2, (ly + ry) // 2)
    # 右上角
    divide_block1 = square(lx, (ly + ry) // 2 + 1, (lx + rx) // 2, ry)
    # 左下角
    divide_block2 = square((lx + rx) // 2 + 1, ly, rx, (ly + ry) // 2)
    # 右下角
    divide_block3 = square((lx + rx) // 2 + 1, (ly + ry) // 2 + 1, rx, ry)
    del square_arr[index_s]
    square_arr.append(divide_block0)
    square_arr.append(divide_block1)
    square_arr.append(divide_block2)
    square_arr.append(divide_block3)

    divide_block.append(divide_block0)
    divide_block.append(divide_block1)
    divide_block.append(divide_block2)
    divide_block.append(divide_block3)

    for i in range(4):
        mat_temp.append(tu[divide_block[i].lx:divide_block[i].rx, divide_block[i].ly:divide_block[i].ry])

    for i in range(4):
        mean.append(np.mean(mat_temp[i]))

    h, w, = mat_temp[0].shape

    for i in range(1, 4):
        mat_temp1.append(cv2.resize(mat_temp[i], (w, h)))
    mat = np.stack((mat_temp[0], mat_temp1[0], mat_temp1[1], mat_temp1[2]))
    return mean, [divide_block0, divide_block1, divide_block2, divide_block3], mat

#得到最终的四叉树
def get_quad_tree(matrix,square_arr):
    eta = 1.1
    st = 1/255.0

    # 第一次分割
    index_0 = 0
    s, divide_block, mat = fenge(index_0,matrix,square_arr)


    #删掉原来的大矩阵，把新的四个子矩阵加进列表
    s_max = max(s)
    s_temp = [i for i in s]
    s_temp_max = max(s_temp) # 求mean中第二大的值
    s_temp.remove(s_temp_max)
    secd_max = max(s_temp)
    index1 = s.index(s_max)
    index2 = square_arr.index(divide_block[index1])
    secd_max_index = s_temp.index(secd_max)
    if (index1 == 0 or index1 == 1) :
        while (True):
            if ((s_max - secd_max) <= st) or (abs(divide_block[index1].lx-divide_block[index1].rx) < 25):
            # if ((s_max - secd_max) <= st):
                break
            s, divide_block, mat = fenge(index2, matrix, square_arr)
            s_max = max(s)
            s_temp = [i for i in s]
            s_temp_max = max(s_temp)  # 求mean中第二大的值
            s_temp.remove(s_temp_max)
            secd_max = max(s_temp)
            index1 = s.index(s_max)
            index2 = square_arr.index(divide_block[index1])
        f_index = index2
        f = mat[index1]
    elif (index1 == 2 or index1 == 3) :
        s = [s[0] * eta, s[1] * eta, s[2], s[3]]
        s_max = max(s)
        s_temp = [i for i in s]
        s_temp_max = max(s_temp)  # 求mean中第二大的值
        s_temp.remove(s_temp_max)
        secd_max = max(s_temp)
        index1 = s.index(s_max)
        index2 = square_arr.index(divide_block[index1])
        secd_max_index = s_temp.index(secd_max)
        while (True):
            if ((s_max - secd_max) <= st) or (abs(divide_block[index1].lx - divide_block[index1].rx) < 25):
            # if ((s_max - secd_max) <= st):
                break
            s, divide_block, mat = fenge(index2, matrix, square_arr)
            s_max = max(s)
            s_temp = [i for i in s]
            s_temp_max = max(s_temp)  # 求mean中第二大的值
            s_temp.remove(s_temp_max)
            secd_max = max(s_temp)
            index1 = s.index(s_max)
            index2 = square_arr.index(divide_block[index1])
        f_index = index2
        f = mat[index1]
    else:
        s_max = max(s)
        s_max2 = s_max
        s_temp = [i for i in s]
        s_temp_max = max(s_temp)  # 求mean中第二大的值
        s_temp.remove(s_temp_max)
        secd_max = max(s_temp)
        secd_max2 = secd_max
        index2 = square_arr.index(divide_block[index1])
        secd_max_index_temp = s.index(secd_max)
        secd_block = divide_block[secd_max_index_temp]
        s_max_temp = s_max
        secd_max_temp = secd_max
        divide_block0 = divide_block
        while (True):
            if ((s_max - secd_max) <= st) or (abs(divide_block0[index1].lx - divide_block0[index1].rx) < 25):
                break
            s, divide_block0, mat = fenge(index2, matrix, square_arr)
            s_max = max(s)
            index1 = s.index(s_max)
            s_temp = [i for i in s]
            s_temp_max = max(s_temp)  # 求mean中第二大的值
            s_temp.remove(s_temp_max)
            secd_max = max(s_temp)
            index2 = square_arr.index(divide_block0[index1])
            f1 = mat[index1]
        index2 = square_arr.index(secd_block)
        index1 = index2
        divide_block1 = divide_block
        secd_max = secd_max_temp
        s_max = s_max_temp
        while (True):
            if ((s_max2 - secd_max2) <= st) or (abs(divide_block1[index1].lx - divide_block1[index1].rx) < 25):
                break
            s, divide_block1, mat = fenge(index2, matrix, square_arr)
            s_max2 = max(s)
            index1 = s.index(s_max2)
            s_temp = [i for i in s]
            s_temp_max = max(s_temp)  # 求mean中第二大的值
            s_temp.remove(s_temp_max)
            secd_max2 = max(s_temp)
            index2 = square_arr.index(divide_block1[index1])
            f2_index = index2
            f2 = mat[index1]

        score1 = np.mean(f1)
        score2 = np.mean(f2)
        if score1 > score2:
            f = f1
            f_index = square_arr.index(divide_block0[index1])
        else:
            f_index = f2_index
            f = f2
    f_temp = f.flatten()
    f_temp = sorted(f_temp, reverse=True)
    if len(f_temp) > 5000:
        f_1 = f_temp[:int(len(f_temp) * 0.1)]
    else:
        f_1 = f_temp[:int(len(f_temp))]
    # f_1 = f_temp[:int(len(f_temp) * 0.1)]
    A = np.mean(f_1)

    return square_arr, A, f_index



def main(img):
    mask_np = img
    # mask_np = resize(mask_np, 224)
    matrix=mask_np
    row=matrix.shape[0]
    col=matrix.shape[1]
    first_square=square(0,0,row-1,col-1)
    square_arr=[first_square]
    square_arr, A, f_index = get_quad_tree(matrix, square_arr)

    print('A:', A)

    return A





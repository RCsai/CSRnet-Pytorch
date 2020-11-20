import numpy as np
import scipy.spatial
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import os
from matplotlib import pyplot as plt


def gaussian_filter_density(img, points):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.
    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.
    return:
    density: the density-map we want. Same shape as input image but only has one channel.
    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape = [img.shape[0], img.shape[1]]
    # print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), "gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)

    # print('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.
        else:
            continue
        if gt_count > 3:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(pt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    # print('done.')
    return density


def one_count(img_path, gt_path):
    filename=img_path.split('/')[-1]
    filenum=filename.split('.')[0]
    img = plt.imread(img_path)
    mat = io.loadmat(gt_path)
    points = mat["image_info"][0][0][0][0][0]
    k = np.zeros((img.shape[0], img.shape[1]))
    k = gaussian_filter_density(img, points)
    people_num = np.sum(k)
    print(filenum+'_num:','\t',people_num)
    return people_num


def many_count(img_root):
    gt_count_txt = './data_test/test_data/result/' + 'gt_count.txt'
    f = open(gt_count_txt, 'w')
    for filename in os.listdir(img_root):
        img_path = img_root + filename
        gt_path = img_path.replace('images', 'gt').replace('IMG', 'GT_IMG').replace('jpg', 'mat')
        gt_name = filename.replace('IMG', 'GT_IMG').split('.')[0]
        people_num = one_count(img_path, gt_path)
        msg = gt_name + ':' + str(people_num) + '\n'
        f.write(msg)
    f.close()
    print('count success')


if __name__ == '__main__':
    img_root = './data_test/test_data/images/'
    many_count(img_root)

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math, cv2, logging, time
from random import random
from PIL import Image, ImageFilter
from mpl_toolkits.mplot3d import Axes3D

def histeq(im, nbr_bins=2048):
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf/(cdf[-1])
    # 使用累积分布函数进行线性插值
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape)

def trans_intensity(im, p):
    temp = im/255
    temp =temp**p
    temp = temp*255
    return temp

def max_min_normal (im):
    maxI = np.max(im)
    minI = np.min(im)
    rangeI = maxI - minI
    im -= minI
    im = im/maxI *255
    return np.asarray(im, np.uint8)

def unit_vec(s,e):
    length = math.sqrt((s[0]-e[0])**2 + (s[1]-e[1])**2)
    unit = ((e[0]-s[0])/length, (e[1]-s[1])/length)
    if unit[0] < 0:
        return (-unit[0], -unit[1])
    else:
        return unit

def getrad(unit_vec):
    return math.pi/2 - math.acos(unit_vec[1])

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    #return cv2.warpAffine(image, M, (nW,nH)), M
    return cv2.warpAffine(image, M, (nW,nH)), M

def get_left_right_width(p1, p2):
    ROUND = lambda x:(int(round(x[0])),int(round(x[1])))
    left = None
    right = None
    width = abs(p1[0] - p2[0])
    if p1[0] < p2 [0]:
        left = p1
        right = p2
    else:
        left = p2
        right = p1
    return ROUND(left), ROUND(right), int(round(width))

def plot3d_image_patch(image_patch):
    height, width  = image_patch.shape
    x = np.arange(0,width,1)
    y = np.arange(0,height,1)
    x, y = np.meshgrid(x, y)
    Z1 = np.ones((height, width))*255
    for y_i in range(0,height):
        for x_i in range(1,width):
            Z1[y_i,x_i] = max_min_normal(trans_intensity(image_patch, 2))[y_i,x_i]

    kernel = np.ones((3,3), np.float32)/9
    after_g = cv2.filter2D(Z1,-1,kernel)        

    ax = Axes3D(plt.figure())
    ax.plot_surface(x, y, after_g,rstride=1, cstride=1, cmap='rainbow')
    ax.set_xlabel('offset_y')
    ax.set_ylabel('offset_length')
    ax.set_zlabel('pixel value')
    plt.show()

def check_seed_point_and_hotmap(hot_map, seedpoint):
    plt.imshow(hot_map,cmap='hot')
    tempax = plt.gca()
    tempax.scatter(seedpoint[0], seedpoint[1])

def get_normalImage_regionPoint(rrimage, regionPoints, lines_datas, normal_size):
    regionPoints = np.asarray(regionPoints, np.float32)
    y_min = -1
    y_max = -1
    for index, x in enumerate(np.sum(rrimage,axis=1)):
        if y_min == -1 and x != 0:
            y_min = index
        elif y_min != -1 and x == 0:
            y_max = index
            break
    x_min = -1
    x_max = -1
    for index, x in enumerate(np.sum(rrimage,axis=0)):
        if x_min == -1 and x != 0:
            x_min = index
        elif x_min != -1 and x == 0:
            x_max = index
            break
    
    # shift and cut
    regionPoints[0,:] -= x_min
    regionPoints[1,:] -= y_min
    
    lines_datas[0,:] -= x_min
    lines_datas[1,:] -= y_min
    
    rrimage = rrimage[y_min:y_max, x_min:x_max]
    
    #resize and scale
    y_scale = normal_size[1]/rrimage.shape[0]
    x_scale = normal_size[0]/rrimage.shape[1]
    
    regionPoints[0,:] *= x_scale
    regionPoints[1,:] *= y_scale
    
    lines_datas[0,:] *= x_scale
    lines_datas[1,:] *= y_scale
    
    regionPoints = np.asanyarray(np.round(regionPoints), np.int32)
    lines_datas = np.asanyarray(np.round(lines_datas), np.int32)
    rrimage = cv2.resize(rrimage, normal_size, interpolation = cv2.INTER_AREA)
    return rrimage, regionPoints, lines_datas

def stack_image_to_color_form(image):
    return np.stack([image]*3, axis=2)

def add_linear_constrain(raw, central_y, loss_func):
    # add constrain to raw
    copy_of_raw = np.copy(raw)
    copy_of_raw = np.asarray(copy_of_raw, dtype=np.int32)
    #print("add_linear_constrain using y:{}".format(central_y))
    height, width = raw.shape
    for y in range(height):
        for x in range(width):
            origin = copy_of_raw[y, x]
            copy_of_raw[y, x] = raw[y, x] + loss_func(abs(y- central_y))
            new = copy_of_raw[y, x]
            if new <origin:
                print('origin is {} and new is {}'.format(origin, new))
                print('error y is {} and central_y is {}'.format(y, central_y))
                print('loss is {}'.format(loss_func(abs(y- central_y))))
    return copy_of_raw

class node():
    def __init__(self,x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return "x:{} y:{}".format(self.x ,self.y)
    def __eq__(self, other):
        if isinstance(other, list):
            for x in other:
                if self.__eq__(x):
                    return True
            return False
        elif isinstance(other, node):
            return self.x == other.x and self.y == other.y
        else:
            raise Exception("try to compare between {} and {}".format("node",type(other)))

class clique:
    def __init__(self, center_point, width, height, image, threadhold):
        self.max_width = width
        self.max_height = height
        self.threadhold = threadhold
        
        self.average_inter = 0
        self.inter_nodes = []
        self.adj_eges = []
        self.adj_nodes = []
        self.image = np.copy(image)
        
        self.inter_nodes.append(center_point)
        self.average_inter = self.image[center_point.y, center_point.x]
        adj_node = self.get_new_adj_node_from_inter_node(center_point)
        for node in adj_node:
            self.adj_nodes.append(node)
            self.adj_eges.append((node,center_point))
    
    def permeation(self):
        available_nodes = self.get_adj_node_lessthreadhold(self.threadhold)
        while available_nodes:
            for node in available_nodes:
                self.add_adj_node_to_inter(node)
            available_nodes = self.get_adj_node_lessthreadhold(self.threadhold)
        #print("final region size is {} pixel".format(len(self.inter_nodes)))
    
    def show_result(self):
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(self.image, cmap='gray')
        image2 = np.zeros_like(self.image)
        for node in self.inter_nodes:
            image2[node.y, node.x] = 255
        ax2.imshow(image2, cmap='gray')
    
    def show_result2(self):
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(self.image, cmap='gray')
        ax1.set_xticks([])
        ax1.set_yticks([])
        #image2 = np.copy(self.image)
        
        ax2.set_xticks([])
        ax2.set_yticks([])
        for node in self.inter_nodes:
            ax1.scatter(node.x, node.y, c='r')
        ax2.imshow(self.image, cmap='gray')
    
    def get_adj_node_lessthreadhold(self, t):
        available_nodes = []
        for adj_node in self.adj_nodes:
            #print(self.image[adj_node.y, adj_node.x])
            if self.image[adj_node.y, adj_node.x] < t:
                available_nodes.append(adj_node)
        return available_nodes
    
    def add_adj_node_to_inter(self, node):
        assert node in self.adj_nodes
        edges = self.find_edge_by_node(node)
        assert edges is not None
        for e in edges:
            self.adj_eges.remove(e)
        self.adj_nodes.remove(node)
        #update adjedge and adjnode and average
        self.average_inter = (self.average_inter * len(self.inter_nodes) + self.image[node.y,node.x])/(len(self.inter_nodes)+1)
        self.inter_nodes.append(node)
        new_adj_nodes = self.get_new_adj_node_from_inter_node(node)
        for new_adjN in new_adj_nodes:
            self.adj_eges.append((node, new_adj_nodes))
            self.adj_nodes.append(new_adjN)
        
    def get_new_adj_node_from_inter_node(self, inter_node):
        assert inter_node in self.inter_nodes
        x, y = inter_node.x, inter_node.y
        temp = []
        if x > 0:
            left = node(x-1, y)
            if left not in self.adj_nodes:
                temp.append(left)
        if x < self.max_width - 1:
            right = node(x+1, y)
            if right not in self.adj_nodes:
                temp.append(right)
        if y > 0:
            up = node(x, y-1)
            if up not in self.adj_nodes:
                temp.append(up)
        if y < self.max_height -1 :
            down = node(x, y+1)
            if down not in self.adj_nodes:
                temp.append(down)
        available_new_adj = []
        for n in temp:
            if n not in self.inter_nodes:
                available_new_adj.append(n)
        return available_new_adj
        
    def find_edge_by_node(self, node):
        edges = []
        for edge in self.adj_eges:
            if node in edge:
                edges.append(edge)
        return edges
    
    def get_homogenous_coordinate(self):
        x = []
        y = []
        for node in self.inter_nodes:
            x.append(node.x)
            y.append(node.y)
        z = [1] * len(x)
        return np.stack([x,y,z],axis=0)
    
def show_result(image,lines,region):
    fig, axs = plt.subplots(1,3)
    for a in axs:
        a.set_xticks([])
        a.set_yticks([])
    image = stack_image_to_color_form(image)
    axs[0].imshow(image)
    #axs[0].set_title('image')
    
    axs[1].imshow(image)
    axs[1].plot(lines[0,:], lines[1,:])
    
    show_rrimage_mask = np.copy(image)
    r = (255,0,0)
    for i in range(len(region[0])):
        show_rrimage_mask[region[1, i],region[0, i]] = r
    axs[2].imshow(show_rrimage_mask)
    #axs[2].set_title('image and regions')
    
def check(lst, re):
    fig, axs = plt.subplots(3,3)
    for index, (image, lines, region) in enumerate(lst):
        image = np.asarray(np.round(image), np.int32)
        image = stack_image_to_color_form(image)
        axs[index,0].imshow(image)
        axs[index,0].set_title('image')
        
        axs[index,1].imshow(image)
        axs[index,1].plot(lines[0,:], lines[1,:])
        if index ==1:
            axs[index,1].add_patch(re)
        axs[index,1].set_title('image and lines')
        
        if region is not None:
            print('region dtype is {}'.format(region.dtype))
            print(len(region[0]))
            show_rrimage_mask = np.copy(image)
            r = (255,0,0)
            for i in range(len(region[0])):
                show_rrimage_mask[region[1, i],region[0, i]] = r
            axs[index,2].imshow(show_rrimage_mask)
            axs[index,2].set_title('image and regions')
    plt.show()

def get_image_and_region(image, lines, flag, line_start_index, last_region, max_start_point):
    if line_start_index >= max_start_point:
        return image, lines, flag, last_region
    
    x, y = lines[0,:], lines[1,:]
    assert len(x)==len(y)
    assert x[line_start_index] > 0
    assert y[line_start_index] > 0
    assert flag == True
    
    
    
    #print('start_index now is {}; max index is {}'.format(str(line_start_index),str(max_start_point)))
    #print('input image shape is {}'.format(image.shape))
    start_point_index = line_start_index
    end_point_index = line_start_index + 1
    start_point_f = lambda x,y:(x[start_point_index], y[start_point_index])
    end_point_f = lambda x,y:(x[end_point_index], y[end_point_index])
        
    s = start_point_f(x,y)
    e = end_point_f(x,y)
    if s[0] == e[0] and s[1]==e[1]:
        return get_image_and_region(image, lines, flag, line_start_index + 2, last_region, max_start_point)
    #if s[0] > e[0]:
    #    s,e = e,s
    rad = getrad (unit_vec(s,e))
    rad = rad*180/math.pi
    #print('rad is {}'.format(rad))
    #rad = -rad
    #print(s,e)
    #print('rorate rad is {}'.format(rad))
    rimage, M = rotate_bound(np.asarray(histeq(image), np.uint8), rad)
    now_x_y = np.stack([x,y,np.ones_like(x)],axis=0)
    trans = np.matmul(M, now_x_y)
        
    x_after_trans = trans[0,:]
    y_after_trans = trans[1,:]
    s_after_trans = start_point_f(x_after_trans, y_after_trans)
    e_after_trans = end_point_f(x_after_trans, y_after_trans)
    if s_after_trans[0] > e_after_trans[0]:
        s_after_trans,e_after_trans = e_after_trans,s_after_trans
    #print('after rad is {}'.format(180/math.pi * getrad (unit_vec(s_after_trans,e_after_trans))))
    after_trans_x_y = np.stack([x_after_trans,y_after_trans,np.ones_like(x)],axis=0)
        
    # get image patch 20 pixel height is enough!
    #print(s_after_trans, e_after_trans)
    left, right, width = get_left_right_width(s_after_trans, e_after_trans)
    #print('width is {}'.format(width))
    #print(left, right)
    height = 20
    rect=patches.Rectangle((left[0],left[1]-height//2),width,height,linewidth=1,edgecolor='r',facecolor='none')
    #print("left x {} left y is {}".format(left[0], left[1]-height//2))
    image_patch = rimage[left[1]-height//2:left[1]+height//2, left[0]:left[0]+width]
    
    #plt.imshow(image_patch, cmap='gray')
    Z1 = histeq(trans_intensity(image_patch, 2))
    loss_func = lambda dist:((dist/4)**3*np.average(Z1) )
    orgin_Z1 = np.copy(Z1)
    Z1 = (add_linear_constrain(Z1, height//2, loss_func))
    Z1 = histeq(Z1)
    
    #print("Z1 shape is {}".format(Z1.shape))
    # do gaussian filter
    kernel = np.ones((3,3), np.float32)/9
    after_gaussian = cv2.filter2D(Z1,-1,kernel) 
    #fig, (ax1,ax2, ax3) = plt.subplots(1,3)
    #ax1.imshow(orgin_Z1, cmap='gray')
    #ax2.imshow(Z1, cmap='gray')
    #ax3.imshow(after_gaussian, cmap='gray')
    #plot3d_image_patch(Z1_with_linear_constrain)
        
        
    # find the lowest seed point
         # find double peaks
    value = np.sum(after_gaussian,axis=1)
    low = np.argmax(value[0:height//2])
    high = np.argmax(value[height//2:height]) + height//2
    #plt.imshow(Z1[low:high+1],cmap='hot')
    Z1Hot = after_gaussian[low:high+1]
        # find seed point between the peeks
    x_seed,y_seed = np.argmax(np.var(Z1Hot,axis=0)), np.argmin(np.average(Z1Hot,axis=1))+low
    #check_seed_point_and_hotmap(Z1Hot, (x_seed, y_seed))
    # do permeate from seed point with given threadhold
        
    # do permeation from seed point to find regions
    thread_hold = np.sqrt(np.max(np.var(Z1Hot,axis=0)))
    #print('thread_hold is {}'.format(thread_hold))
    cli = clique(node(x_seed, y_seed), width,height,after_gaussian,thread_hold)
    cli.permeation()
    #cli.show_result2()
    image_patch_coor = cli.get_homogenous_coordinate()
        
    rimage_coor = np.copy(image_patch_coor)
    
    #print("left x {} left y is {}".format(left[0], left[1]-height//2))
    rimage_coor[0,:] = image_patch_coor[0,:] + left[0] 
    rimage_coor[1,:] = image_patch_coor[1,:] + left[1]-height//2
    r_regin_points = np.copy(rimage_coor)
    # rotate_to_original
    rrimage, RM = rotate_bound(rimage, -rad)
        
    rebuild_x_y = np.matmul(RM, after_trans_x_y)
    regin_points = np.asarray(
                            np.around(
                                np.matmul(RM, rimage_coor)
                                    ), np.int32
                                )
    
    rrimage, rregin_points, rebuild_x_y = get_normalImage_regionPoint(rrimage, regin_points, rebuild_x_y, size)
    if last_region is not None:
        rregin_points = np.concatenate([last_region, rregin_points], axis=1)
    #check
    #lst = [(image , lines, last_region), 
    #       (rimage, after_trans_x_y, r_regin_points), 
    #       (rrimage, rebuild_x_y, rregin_points)]
    #check(lst,rect)
    return get_image_and_region(rrimage, rebuild_x_y, flag, line_start_index + 2, rregin_points, max_start_point)


sub_folder = 'validate'
root = './data/vgg_19_data/'+ sub_folder +'/numpy_form_record_file_224_224/'
N = lambda filename:os.path.join(root, filename)
images_np = np.load(N('raw_data.npy'))
flages_np = np.load((N('flag_data.npy')))
lines_np = np.load(N('line_data.npy'))

save_root = './data/vgg_19_data/'+ sub_folder +'/numpy_form_record_file_224_224/line_mask'
SaveName = lambda filename:os.path.join(save_root, filename)
if not os.path.isdir(save_root):
    os.mkdir(save_root)

size = images_np.shape[1:3]


new_image = []
mask_image = []
count = 1

neg_images = []

for i in range(images_np.shape[0]):
    #i =4862
    flag = flages_np[i]
    if not flag:
        neg_images.append(images_np[i].reshape(224,224))
        continue
    print('deal image {} index is {}'.format(count,i))
    count +=1
    image = images_np[i]
    lines = lines_np[i]
    

    x, y = lines[0:np.argwhere(lines == -1)[0][0]:2], lines[1:np.argwhere(lines == -1)[0][0]:2]
    lines_x_y = np.stack([x,y],axis=0)
    max_start_point = np.where(lines==-1)[0][0]/2

    image_final, lines_final, flag_final, region_point = get_image_and_region(histeq(image.reshape(224,224)), lines_x_y, flag, 0 ,None, max_start_point)
    if region_point is None:
        continue
    new_image.append(image_final)
    mask = np.zeros_like(image_final)

    for rp_index in range(len(region_point[0])):
        mask[region_point[1,rp_index], region_point[0,rp_index]] = 1
    mask_image.append(mask)

image_pics = np.stack(new_image, axis=0)
mask_pics = np.stack(mask_image, axis=0)


neg_pics = np.stack(map(histeq, neg_images), axis=0)
np.save(SaveName('images.npy'), image_pics)
np.save(SaveName('mask.npy'), mask_pics)
np.save(SaveName('neg_image.npy'), neg_pics)
    
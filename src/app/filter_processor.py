import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from xml.etree import ElementTree as ET
import os
import torch
import cv2
import pandas as pd
import csv

def load_filter_points(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        # points = np.empty((0,2), float)
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = float(row[1]), float(row[2])
                # print(x, y)
                points[row[0]] = (x, y)
                # points = np.append(points, np.array([[x, y]]), axis=0)
            except ValueError:
                continue
        return points

def calculate_affine_matrix_for_2_points(src, dst):
    sin60 = np.sin(np.pi/3)
    cos60 = np.cos(np.pi/3)

    ## Add 1 point to src and dst that make equilateral triangle
    x_src = src[0][0] + (src[1][0] - src[0][0]) * cos60 - (src[1][1] - src[0][1]) * sin60
    y_src = src[0][1] + (src[1][0] - src[0][0]) * sin60 + (src[1][1] - src[0][1]) * cos60
    
    x_dst = dst[0][0] + (dst[1][0] - dst[0][0]) * cos60 - (dst[1][1] - dst[0][1]) * sin60
    y_dst = dst[0][1] + (dst[1][0] - dst[0][0]) * sin60 - (dst[1][1] - dst[0][1]) * cos60

    inp = np.append(src, np.array([[x_src, y_src]]), axis=0)
    out = np.append(dst, np.array([[x_dst, y_dst]]), axis=0)

    affine_matrix = cv2.estimateAffinePartial2D(inp, out)
    return affine_matrix[0]

##Return vector Triangle and of vector Id
def find_delauney(img, points):
    size = img.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    subdiv.initDelaunay(rect)
    # for p in points:
    #     subdiv.insert(p)

    # vertex_dict = {}
    for index in range(points.shape[0]):
        # print(points[index])
        # vertex_dict[str(points[index][0]) + '.' + str(points[index][1])] = index
        subdiv.insert(points[index])

    delauney_id = []
    triangleList = subdiv.getTriangleList()
    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        pt = np.float32(pt)
        ind = []
        for j in range(0, 3):
            for k in range(0, len(points)):
                if (pt[j][0] == points[k][0]) and (pt[j][1] == points[k][1]):
                    ind.append(k)
                    break
                    # ind.append(vertex_dict[str(pt[j][0]) + '.' + str(pt[j][1])])

        if len(ind) == 3:
            delauney_id.append((ind[0], ind[1], ind[2]))
        if len(ind) != 3:
            print("Error: ", len(ind))
    return triangleList, delauney_id

def draw_delaunay(img, points):

    triangleList, triangleId = find_delauney(img, points)
    # print(len(triangleId))
    for t in triangleList :
         
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        ##Draw triagle from 3 points using plt
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
        plt.plot([pt2[0], pt3[0]], [pt2[1], pt3[1]])
        plt.plot([pt3[0], pt1[0]], [pt3[1], pt1[1]])
    
    plt.imshow(img)

def wrap_affine(src, dst, tria1, tria2):
    rect1 = cv2.boundingRect(np.float32([tria1]))
    rect2 = cv2.boundingRect(np.float32([tria2]))

    tria1Cropped = []
    tria2Cropped = []

    for i in range(3):
        tria1Cropped.append(((tria1[i][0] - rect1[0]), (tria1[i][1] - rect1[1]))) #x - x0, y - y0
        tria2Cropped.append(((tria2[i][0] - rect2[0]), (tria2[i][1] - rect2[1])))
    
    srcCropped = src[rect1[1]:rect1[1]+rect1[3], rect1[0]:rect1[0]+rect1[2]] #y -> y + h, x -> x + w
    
    affine_matrix = cv2.getAffineTransform(np.float32(tria1Cropped), np.float32(tria2Cropped))
    mask = np.zeros((rect2[3], rect2[2], 3), dtype=np.float32)
    mask = cv2.fillConvexPoly(mask, np.int32(tria1Cropped), (1.0, 1.0, 1.0), 16, 0)
    
    dstCropped = cv2.warpAffine(srcCropped, affine_matrix, (rect2[2], rect2[3]), None, 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    dstCropped = dstCropped * mask
    dst[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]] *= ((1.0, 1.0, 1.0) - mask)
    dst[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]] += dstCropped
    return dst

def apply_filter(img, img_points, filter_img, filter_points, mask_filter):
    # res = img.copy()
    triangleList1, triagleId1 = find_delauney(img, img_points)
    # triangleList2, triagleId2 = find_delauney(filter_img, filter_points)
    number_of_triangles = len(triagleId1)

    for i in range(0, number_of_triangles):

        tria1 = np.array(img_points[np.array(triagleId1[i])], dtype=np.float32)
        tria2 = np.array(filter_points[np.array(triagleId1[i])], dtype=np.float32)

        r1 = cv2.boundingRect(tria1) #x, y, w, h
        r2 = cv2.boundingRect(tria2) #x, y, w, h

        trig1Cropped = []
        trig2Cropped = []

        for i in range(3):
            trig1Cropped.append(((tria1[i][0] - r1[0]),(tria1[i][1] - r1[1]))) #x - x0, y - y0
            trig2Cropped.append(((tria2[i][0] - r2[0]),(tria2[i][1] - r2[1]))) #x - x0, y - y0
        
        img2Cropped = filter_img[r2[1]: r2[1] + r2[3], r2[0]: r2[0] + r2[2]] #y -> y + h, x -> x + w
        mask_filterCropped = mask_filter[r2[1]:r2[1] + r2[3] , r2[0]:r2[0 ] + r2[2]]

        # print(mask_filterCropped)

        warpMat = cv2.getAffineTransform( np.float32(trig2Cropped), np.float32(trig1Cropped) )

        img1Cropped = cv2.warpAffine( img2Cropped, warpMat, (r1[2], r1[3]), None, 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT )
        img1Cropped = np.nan_to_num(img1Cropped, nan = 0)
        # print(img1Cropped)

        mask_filterCropped = cv2.warpAffine( mask_filterCropped, warpMat, (r1[2], r1[3]), None, 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT )
        
        mask_filterCropped = cv2.merge((mask_filterCropped, mask_filterCropped, mask_filterCropped))

        mask = np.zeros((r1[3], r1[2], 3), dtype = np.float32)
        mask = cv2.fillConvexPoly(mask, np.int32(trig1Cropped), (1.0, 1.0, 1.0), 16, 0)

        # print(mask_filterCropped)
        # MAX = np.max(mask_filterCropped)
        mask_filterCropped = np.array(mask_filterCropped * (1./255.), dtype = np.float32)
        # print(mask_filterCropped)
        
        mask = np.multiply(mask, mask_filterCropped)

        img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]] = img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]] * ((1.0, 1.0, 1.0) - mask)
        img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]] = img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]] + img1Cropped[:,:,:] * mask

    img = cv2.GaussianBlur(img, (3, 3), 10)

    return img
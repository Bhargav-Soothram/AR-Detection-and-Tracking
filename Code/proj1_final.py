from cv2 import normalize
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.fft as ft
import os
import copy


# Creates a circular mask
def generate_mask(size, radius):
    height, width = size
    center = [int(height/2), int(width/2)]
    X, Y = np.ogrid[:height, :width]
    mask_area = (X - center[0]) ** 2 + (Y - center[1]) ** 2 <= radius*radius
    mask = np.ones((height, width)) 
    mask[mask_area] = 0
    return mask


# Returns the edge-image using Fourier Transform  
def get_edges(image, save_path=None, save=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thrshld_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    median_blur = cv2.medianBlur(thrshld_img,5)
    ft_img = ft.fft2(median_blur, axes=(0, 1))
    ft_shiftd = ft.fftshift(ft_img)
    magn_ft_shiftd = 20*np.log(np.abs(ft_shiftd))

    # masking the image
    cmask = generate_mask(gray.shape, 100)
    ft_MaskingforEdges = ft_shiftd * cmask
    mag_masked = 20*np.log(np.abs(ft_MaskingforEdges))

    #retrieving the image
    rtrv_shft_img = ft.ifftshift(ft_MaskingforEdges)
    rtrv_img_raw = ft.ifft2(rtrv_shft_img)
    rtrv_img = np.uint8(np.abs(rtrv_img_raw))

    f, axarr = plt.subplots(2, 2, figsize=(15, 15))
    axarr[0, 0].imshow(gray, cmap = 'gray')
    axarr[0, 0].set_title('Selected Frame')
    axarr[0, 0].axis('off')

    axarr[0, 1].imshow(magn_ft_shiftd, cmap = 'gray')
    axarr[0, 1].set_title('Fourier Magnitude')
    axarr[0, 1].axis('off')

    axarr[1, 0].imshow(mag_masked, cmap = 'gray')
    axarr[1, 0].set_title('Fourier Magnitude with mask')
    axarr[1, 0].axis('off')

    axarr[1, 1].imshow(rtrv_img, cmap = 'gray')
    axarr[1, 1].set_title('Edge Image')
    axarr[1, 1].axis('off')
    # plt.show()
    if save == True:
        plt.savefig(save_path + 'Edges.jpg')
    return rtrv_img


# Returns the corners of the image in the scene
def get_corners(an_image):
    image_copy = copy.deepcopy(an_image)
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    blurd_image = cv2.medianBlur(gray_image, ksize=7)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(blurd_image, cv2.MORPH_CLOSE, kernel)
    corners = cv2.goodFeaturesToTrack(thresh, 50, 0.28, 7)
    my_corners = list()
    for pt in corners:
        x = int(pt[0][0])
        y = int(pt[0][1])
        my_corners.append((x, y))   
    return my_corners


# Returns the corners of the tag from the image
def get_tagCorners(corner_list):
        max_idx = np.argmax(corner_list, axis=0)
        min_idx = np.argmin(corner_list, axis=0)
        sorted_crnrs = list([corner_list[max_idx[0]], corner_list[max_idx[1]], corner_list[min_idx[0]], corner_list[min_idx[1]]])
        updated_list = [crnr for crnr in corner_list if crnr not in sorted_crnrs]
        max_idx = np.argmax(updated_list, axis=0)
        min_idx = np.argmin(updated_list, axis=0)
        tag_crnrs = list([updated_list[min_idx[0]], updated_list[min_idx[1]], updated_list[max_idx[0]], updated_list[max_idx[1]]])
        # print(tag_crnrs)
        return tag_crnrs


# Returns the tag orientation array of 4 values 
def get_TagOrientation(tag_image):
    tag_image = cv2.cvtColor(tag_image, cv2.COLOR_BGR2GRAY)
    ret,tag = cv2.threshold(np.uint8(tag_image), 230 ,255,cv2.THRESH_BINARY)
    tag_size = tag.shape[0]
    threshold = 0.7*(tag_size*tag_size)
    grid_size = 8
    pixels_per_grid = int(tag_size/grid_size)
    readable_tag = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            selection = tag[i*pixels_per_grid:(i+1)*pixels_per_grid, j*pixels_per_grid:(j+1)*pixels_per_grid]
            if np.sum(selection) > threshold and np.median(selection) == 255:
                readable_tag[i, j] = 255
    ar_tag = readable_tag[2:6, 2:6]
    tag_wdth = len(ar_tag)
    tag_blocks = [ar_tag[0, 0], ar_tag[0, tag_wdth-1], ar_tag[tag_wdth-1, 0], ar_tag[tag_wdth-1, tag_wdth-1]]
    # if tag_blocks[0] == 255:
    #     print('180!')
    # if tag_blocks[1] == 255:
    #     print('270!')
    # if tag_blocks[2] == 255:
    #     print('90!')
    # if tag_blocks[3] == 255:
    #     print('Upright!')
    return tag_blocks


# Computes and returns the homography given two sets of corner points 
def get_homography(crnrs_img1, crnrs_img2):
    try:
        A = list()
        for i in range(len(crnrs_img1)):
            x, y = crnrs_img1[i][0], crnrs_img1[i][1]
            xp1, yp1 = crnrs_img2[i][0], crnrs_img2[i][1]
            A.append(list([x, y, 1, 0, 0, 0, -xp1*x, -xp1*y, -xp1]))
            A.append(list([0, 0, 0, x, y, 1, -yp1*x, -yp1*y, -yp1]))
        A= np.asarray(A)
        _, _, V = np.linalg.svd(A)
        L = V[-1, :]/V[-1, -1]
        H = L.reshape(3, 3)
        return H
    except:
        pass


# Gets the inverse transform of the 
def warpPerspective(H, img, tag_length, tag_width):
    H_inv=np.linalg.inv(H)
    warped=np.zeros((tag_length,tag_width,3),np.uint8)
    for a in range(tag_length):
        for b in range(tag_width):
            f = [a,b,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            try:
                warped[a][b] = img[int(y/z)][int(x/z)]
            except:
                warped[a][b] = 255
    return warped
    

# Rotates the frame corners by 90 degrees each time when used
def get_RotatedCorners(patch_corners):
    crnrs = list(patch_corners.copy())
    direction_sq = crnrs.pop(-1)
    crnrs.insert(0, direction_sq)
    return np.array(crnrs)


# Maps the testudo image over the tag 
def map_testudo_over_img(frame, tag_crnrs, testudo_image):
    try:
        plane_points = [(160, 0), (0, 0), (0, 160), (160, 160)]
        hmgrphy = get_homography(tag_crnrs, plane_points)
        testudo_image = cv2.resize(testudo_image, (160, 160))
        for a in range(160):
            for b in range(160):
                f = [a, b, 1]
                f = np.reshape(f, (3, 1))
                x, y, z = np.dot(np.linalg.inv(hmgrphy), f)
                frame[int(y/z)][int(x/z)] = testudo_image[a][b]
        return frame
    except:
        return frame


# Constructing projection matrix P
def get_ProjectionMtrx(homography_mtrx, calibration_mtrx):
    B_tilde = np.dot((np.linalg.inv(calibration_mtrx)), homography_mtrx)
    if np.linalg.norm(B_tilde) > 0:
        B = B_tilde
    else:
        B = -1 * B_tilde
    lamda = 2/(np.linalg.norm(B[:, 0])+np.linalg.norm(B[:, 1]))
    r1 = lamda*B[:, 0]
    r2 = lamda*B[:, 1]
    r3 = np.cross(r1, r2)
    t = lamda*B[:, 2]
    projection_mtrx = np.column_stack((r1, r2, r3, t))
    projection_mtrx = np.dot(calibration_mtrx, projection_mtrx)
    projection_mtrx = projection_mtrx / projection_mtrx[2,3]
    return projection_mtrx


# Applies the projection matrix to the tag corners and returns the corresponding projections
def get_ProjectedPoints(pts, prj_mtrx):
    stacked = np.column_stack((pts[:, 0], pts[:, 1], pts[:, 2], np.ones(pts[:, 0].size))).T
    raw_prj = prj_mtrx.dot(stacked)
    nrmlzd_pts = raw_prj/raw_prj[2, :]
    prjctd_pts = nrmlzd_pts[0:2, :].T.astype(int)
    return prjctd_pts


# This function returns cube vertices when the planar(2D) coordinates are given 
def get_cube(coords2d, height):
    cube_pts = list()
    for cd in coords2d:
        x = cd[0]
        y = cd[1]
        z = -height
        cube_pts.append((x, y, z))
    cube_pts = np.stack(cube_pts)
    return cube_pts


# Function to draw the edges of the cube on the image
def draw_cube(image, base_coords, projected_coords):
    x1, x2, x3, x4 = base_coords[0][0], base_coords[1][0], base_coords[2][0], base_coords[3][0]
    y1, y2, y3, y4 = base_coords[0][1], base_coords[1][1], base_coords[2][1], base_coords[3][1]
    x1p, x2p, x3p, x4p = projected_coords[0][0], projected_coords[1][0], projected_coords[2][0], projected_coords[3][0]
    y1p, y2p, y3p, y4p = projected_coords[0][1], projected_coords[1][1], projected_coords[2][1], projected_coords[3][1]

    color = (0, 0, 255)
    image = cv2.line(image, (x1, y1), (x2, y2), color=color, thickness=3)
    image = cv2.line(image, (x2, y2), (x3, y3), color=color, thickness=3)
    image = cv2.line(image, (x3, y3), (x4, y4), color=color, thickness=3)
    image = cv2.line(image, (x4, y4), (x1, y1), color=color, thickness=3)

    image = cv2.line(image, (x1, y1), (x1p, y1p), color=color, thickness=3)
    image = cv2.line(image, (x2, y2), (x2p, y2p), color=color, thickness=3)
    image = cv2.line(image, (x3, y3), (x3p, y3p), color=color, thickness=3)
    image = cv2.line(image, (x4, y4), (x4p, y4p), color=color, thickness=3)

    image = cv2.line(image, (x1p, y1p), (x2p, y2p), color=color, thickness=3)
    image = cv2.line(image, (x2p, y2p), (x3p, y3p), color=color, thickness=3)
    image = cv2.line(image, (x3p, y3p), (x4p, y4p), color=color, thickness=3)
    image = cv2.line(image, (x4p, y4p), (x1p, y1p), color=color, thickness=3)
    return image


tag_size = 160
plane_points = [(tag_size-1, 0), (0, 0), (0, tag_size-1), (tag_size-1, tag_size-1)]
K = np.array([[1346.100595, 0, 932.1633975], [0, 1355.933136, 654.8986796], [0, 0, 1]])   # Camera calibration matrix 
testudo_img = cv2.imread('testudo.png')   # Loading the Testudo image


testudo = int(input("Enter 1 to see Testudo on AR tag or 0 to see cube on AR tag"))    # Change this to False to place the cube on the AR tag
video_path = "1tagvideo.mp4"
ar_track1 = cv2.VideoCapture(video_path)  # VideoCapture object
disp_duration = 1
all_frames = list()
if not ar_track1.isOpened():
    print("Error opening video stream or file!")
count = 0
while (ar_track1.isOpened()):
    ret, frame = ar_track1.read()
    if ret:
        count += 1
        print("Evaluating frame ", count)
        all_frames.append(frame)
        corners = get_corners(frame)
        tag_corners = get_tagCorners(corners)
        H_tp = get_homography(np.float32(tag_corners), np.float32(plane_points))
        warpd_img = warpPerspective(H_tp, frame, tag_size, tag_size)
        to_rotate = 0
        orientation = get_TagOrientation(warpd_img)
        if orientation[0] == 255:
            to_rotate = 180
        if orientation[1] == 255:
            to_rotate = 270
        if orientation[2] == 255:
            to_rotate = 90
        if orientation[3] == 255:
            to_rotate = 0

        while to_rotate != 0:
            tag_corners = get_RotatedCorners(tag_corners)
            to_rotate -= 90
        
        if testudo == True:
            testdo_img = map_testudo_over_img(frame, tag_corners, testudo_img)
            cv2.imshow('frame', testdo_img)
        else:
            H_pt = get_homography(np.float32(plane_points), np.float32(tag_corners))
            P = get_ProjectionMtrx(H_pt, K)
            cube_vertices = get_cube(plane_points, tag_size-1)
            projected_vertices = get_ProjectedPoints(cube_vertices, P)
            cube_on_img = draw_cube(frame, base_coords=tag_corners, projected_coords=projected_vertices)
            cv2.imshow('frame', cube_on_img)
        if cv2.waitKey(disp_duration) & 0xFF == ord('q'):
            break
    else:
        break


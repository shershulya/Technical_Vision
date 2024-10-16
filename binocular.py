import cv2
import numpy as np
import time

# start_time = time.time()

class Rectangle:
    def __init__(self, left_bottom, left_top, right_bottom, right_top):
        self.left_bottom = left_bottom
        self.left_top = left_top
        self.right_bottom = right_bottom
        self.right_top = right_top
    def print(self):
        print(self.left_bottom,
              self.left_top,
              self.right_bottom,
              self.right_top,)

def range1(start, end):
    return range(start, end + 1)

def GenerateRectanglePoints(left_bottom, left_top, right_bottom, right_top):
    points = []
    z = left_bottom[2]

    y = left_bottom[1]
    for x in range1(left_bottom[0], right_bottom[0]):
        points.append((x, y, z))
    y = left_top[1]
    for x in range1(left_top[0], right_top[0]):
        points.append((x, y, z))
    x = left_bottom[0]
    for y in range1(left_bottom[1], left_top[1]):
        points.append((x, y, z))
    x = right_bottom[0]
    for y in range1(right_bottom[1], right_top[1]):
        points.append((x, y, z))
    return points

def GenerateBackRectangles(z, left_bottom_x, left_bottom_y, right_top_x, right_top_y, max_size, number):
    rectangles = []
    for _ in range(number):
        height = np.random.randint(0, max_size)
        width = np.random.randint(0, max_size)
        start_x = np.random.randint(left_bottom_x, right_top_x - width)
        start_y = np.random.randint(left_bottom_y, right_top_y - height)
        rectangles.append(Rectangle((start_x, start_y, z),
                                    (start_x, start_y + height, z),
                                    (start_x + width, start_y, z),
                                    (start_x + width, start_y + height, z)))
    return rectangles

def GenerateFrontRectangles(z,
                            left_bottom_x,
                            left_bottom_y,
                            right_top_x,
                            right_top_y,
                            hole_lb_x,
                            hole_lb_y,
                            hole_rt_x,
                            hole_rt_y,
                            max_size, number):
    rectangles = []
    i = 0
    while i < number:
        height = np.random.randint(0, max_size)
        width = np.random.randint(0, max_size)
        start_x = np.random.randint(left_bottom_x, right_top_x - width)
        start_y = np.random.randint(left_bottom_y, right_top_y - height)

        if (start_x >= hole_lb_x and start_x <= hole_rt_x and
            start_y >= hole_lb_y and start_y <= hole_rt_y):
            continue
        if (start_x >= hole_lb_x and start_x <= hole_rt_x and
            start_y + height >= hole_lb_y and start_y + height <= hole_rt_y):
            continue
        if (start_x + width >= hole_lb_x and start_x + width <= hole_rt_x and
            start_y >= hole_lb_y and start_y <= hole_rt_y):
            continue
        if (start_x + width >= hole_lb_x and start_x + width <= hole_rt_x and
            start_y + height >= hole_lb_y and start_y + height <= hole_rt_y):
            continue
        
        rectangles.append(Rectangle((start_x, start_y, z),
                                    (start_x, start_y + height, z),
                                    (start_x + width, start_y, z),
                                    (start_x + width, start_y + height, z)))
        i += 1
    return rectangles



def depth_map_opencv(img_left, img_right, sigma=100):
    window_size = 3
    lmbda = 800

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=1*16,
        blockSize=window_size,
        P1=8 * 3 * window_size,
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        preFilterCap=63,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    disparity_left = left_matcher.compute(img_left, img_right)
    disparity_right = right_matcher.compute(img_right, img_left)
    disparity_left = np.int16(disparity_left)
    disparity_right = np.int16(disparity_right)
    img_filtered = wls_filter.filter(disparity_left, img_left, None, disparity_right)

    img_filtered = cv2.normalize(src=img_filtered, dst=img_filtered, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    img_filtered = np.uint8(img_filtered)

    return img_filtered

def calc_disparity(image_left, image_right, window_size, search_range):
    start_time = time.time()

    array_left = np.array(image_left, dtype='int64')
    array_right = np.array(image_right, dtype='int64')
    disp_matrix = []

    right_border = array_left.shape[0] - window_size

    for row in range(len(array_left)):
        if row % 10 == 0:
            print(f"Disparity calculated for {row} rows.")
        disps = []
        for col1 in range(right_border):
            win1 = array_left[row, col1:col1 + window_size]
            if col1 + search_range < right_border:
                search = search_range
            else:
                search = right_border - col1
            sads = []

            for col2 in range(col1, col1 + search):
                win2 = array_right[row, col2:col2 + window_size]

                sad = np.sum(np.abs(np.subtract(win1, win2)))
                sads.append(sad)

            disparity = np.argmin(sads)
            disps.append(disparity)
        disp_matrix.append(disps)
     
    disp_matrix = np.array(disp_matrix)
    end_time = time.time()
    print(f"Time elapsed during disparity calculations: {int(end_time - start_time)}s")
    return disp_matrix

if __name__ == '__main__':
    # Task input
    focus = 5
    camera_l_coord = (500, 500, 0)
    cameras_dist = 100
    planes_dist = 1
    camera_plane_dist = 9
    camera_plane_x = 1
    camera_plane_y = 4
    plane_front_size = 900
    plane_back_size = 750
    
    plane_front_lb = (camera_l_coord[0] + camera_plane_x,
                      camera_l_coord[1] - camera_plane_y,
                      camera_l_coord[2] + camera_plane_dist)
    
    plane_front_lt = (plane_front_lb[0],
                      plane_front_lb[1] + plane_front_size,
                      plane_front_lb[2])

    plane_front_rb = (plane_front_lb[0] + plane_front_size,
                      plane_front_lb[1],
                      plane_front_lb[2])

    plane_front_rt = (plane_front_lb[0] + plane_front_size,
                      plane_front_lb[1] + plane_front_size,
                      plane_front_lb[2])

    hole_size = plane_front_size // 2

    hole_lb = (plane_front_lb[0] + hole_size // 2,
               plane_front_lb[1] + hole_size // 2,
               plane_front_lb[2])
    
    hole_lt = (hole_lb[0],
               hole_lb[1] + hole_size,
               hole_lb[2])

    hole_rb = (hole_lb[0] + hole_size,
               hole_lb[1],
               hole_lb[2])

    hole_rt = (hole_lb[0] + hole_size,
               hole_lb[1] + hole_size,
               hole_lb[2])
    
    offset = (plane_front_size - plane_back_size) // 2
    plane_back_lb = (plane_front_lb[0] + offset,
                     plane_front_lb[1] + offset,
                     plane_front_lb[2] + planes_dist)
    
    plane_back_lt = (plane_front_lt[0] + offset,
                     plane_front_lt[1] - offset,
                     plane_front_lt[2] + planes_dist)

    plane_back_rb = (plane_front_rb[0] - offset,
                     plane_front_rb[1] + offset,
                     plane_front_rb[2] + planes_dist)

    plane_back_rt = (plane_front_rt[0] - offset,
                     plane_front_rt[1] - offset,
                     plane_front_rt[2] + planes_dist)
    
    plane_front_points = GenerateRectanglePoints(plane_front_lb,
                                                 plane_front_lt,
                                                 plane_front_rb,
                                                 plane_front_rt)

    hole_points = GenerateRectanglePoints(hole_lb,
                                          hole_lt,
                                          hole_rb,
                                          hole_rt)

    plane_back_points = GenerateRectanglePoints(plane_back_lb,
                                                plane_back_lt,
                                                plane_back_rb,
                                                plane_back_rt)
###############################################
    img_left = np.zeros((1000, 1000))
    img_right = np.zeros((1000, 1000))

    for point in plane_front_points:
        u = focus * point[0] / point[2]
        v = focus * point[1] / point[2]
        img_left[int(u)][int(v)] = 255
        ###############################################
        u = focus * point[0] / point[2]
        v = focus * (point[1] + cameras_dist) / point[2]
        img_right[int(u)][int(v)] = 255

    # for point in hole_points:
    #     u = focus * point[0] / point[2]
    #     v = focus * point[1] / point[2]
    #     img_left[int(u)][int(v)] = 255
    #     ###############################################
    #     u = focus * point[0] / point[2]
    #     v = focus * (point[1] + cameras_dist) / point[2]
    #     img_right[int(u)][int(v)] = 255

    # for point in plane_back_points:
    #     u = focus * point[0] / point[2]
    #     v = focus * point[1] / point[2]
    #     img_left[int(u)][int(v)] = 255
    #     ###############################################
    #     u = focus * point[0] / point[2]
    #     v = focus * (point[1] + cameras_dist) / point[2]
    #     img_right[int(u)][int(v)] = 255


###############################################
###############################################

    rectangles_font = GenerateFrontRectangles(plane_front_lb[2],
                                              plane_front_lb[0],
                                              plane_front_lb[1],
                                              plane_front_rt[0],
                                              plane_front_rt[1],
                                              hole_lb[0],
                                              hole_lb[1],
                                              hole_rt[0],
                                              hole_rt[1],
                                              20, 1500)

    for rect in rectangles_font:
        rect_points = GenerateRectanglePoints(rect.left_bottom,
                                              rect.left_top,
                                              rect.right_bottom,
                                              rect.right_top)
        for point in rect_points:
            u = int(focus * point[0] / point[2])
            v = int(focus * point[1] / point[2])
            img_left[u][v] = 255
            ###############################################
            u = int(focus * point[0] / point[2])
            v = int(focus * (point[1] + cameras_dist) / point[2])
            img_right[u][v] = 255
    
    rectangles_back = GenerateBackRectangles(plane_back_lb[2],
                                             plane_back_lb[0],
                                             plane_back_lb[1],
                                             plane_back_rt[0],
                                             plane_back_rt[1],
                                             20, 1500)

    hole_prj_left_lb_x = int(focus * hole_lb[0] / hole_lb[2])
    hole_prj_left_lb_y = int(focus * hole_lb[1] / hole_lb[2])
    hole_prj_left_rt_x = int(focus * hole_rt[0] / hole_rt[2])
    hole_prj_left_rt_y = int(focus * hole_rt[1] / hole_rt[2])

    hole_prj_right_lb_x = int(focus * hole_lb[0] / hole_lb[2])
    hole_prj_right_lb_y = int(focus * (hole_lb[1] + cameras_dist) / hole_lb[2])
    hole_prj_right_rt_x = int(focus * hole_rt[0] / hole_rt[2])
    hole_prj_right_rt_y = int(focus * (hole_rt[1] + cameras_dist) / hole_rt[2])

    for rect in rectangles_back:
        rect_points = GenerateRectanglePoints(rect.left_bottom,
                                              rect.left_top,
                                              rect.right_bottom,
                                              rect.right_top)
        for point in rect_points:
            u = int(focus * point[0] / point[2])
            v = int(focus * point[1] / point[2])
            if (u >= hole_prj_left_lb_x and
                u <= hole_prj_left_rt_x and
                v >= hole_prj_left_lb_y and
                v <= hole_prj_left_rt_y):
                img_left[u][v] = 255
            ###############################################
            u = int(focus * point[0] / point[2])
            v = int(focus * (point[1] + cameras_dist) / point[2])
            if (u >= hole_prj_right_lb_x and
                u <= hole_prj_right_rt_x and
                v >= hole_prj_right_lb_y and
                v <= hole_prj_right_rt_y):
                img_right[u][v] = 255
###############################################
    img_left = np.array(img_left)
    cv2.imwrite('left_image.png', img_left)
    image_left = cv2.imread('left_image.png')

    # cv2.namedWindow("left_image", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("left_image", 1000, 1000)
    # cv2.moveWindow("left_image", 0, 10)
    # cv2.imshow('left_image', image_left)
###############################################
    img_right = np.array(img_right)
    cv2.imwrite('right_image.png', img_right)
    image_right = cv2.imread('right_image.png')

    # cv2.namedWindow("right_image", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("right_image", 1000, 1000)
    # cv2.moveWindow("right_image", 1000, 0)
    # cv2.imshow('right_image', image_right)
###############################################
    img_stereo_pair = np.uint8(image_left / 2 + image_right / 2)
    cv2.imwrite('stereo_pair.png', img_stereo_pair)
    image_stereo_pair = cv2.imread('stereo_pair.png')

    # cv2.namedWindow("stereo_pair", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("stereo_pair", 1000, 1000)
    # cv2.moveWindow("stereo_pair", 500, 10)
    # cv2.imshow('stereo_pair', img_stereo_pair)
###############################################
    window_size = 20
    search_range = 50

    # image_left = cv2.imread('camera/camera_left.png')
    # image_right = cv2.imread('camera/camera_right.png')
    
    dim = (250, 250)
    image_left = cv2.resize(image_left, dim)
    image_right = cv2.resize(image_right, dim)


    image_depth_map = calc_disparity(image_left, image_right, window_size, search_range)


    print(image_depth_map.dtype, image_depth_map.shape)
    print('max value: ', np.amax(image_depth_map))
    print('min value: ', np.amin(image_depth_map))

    image_depth_map = cv2.applyColorMap((image_depth_map * (256 / np.amax(image_depth_map))).astype(np.uint8), cv2.COLORMAP_BONE)

    # image_depth_map = depth_map_opencv(image_left, image_right)
    image_depth_map = cv2.resize(image_depth_map, (1000, 1000))
    cv2.imwrite('depth_map.png', image_depth_map)

    cv2.namedWindow("depth_map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("depth_map", 1000, 1000)
    cv2.moveWindow("depth_map", 500, 10)
    cv2.imshow('depth_map', image_depth_map)
###############################################

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print("--- %s seconds ---" % (time.time() - start_time))

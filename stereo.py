from PIL import Image
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import time

class Process:
    def __init__(self, image):
        self.image = Image.open(image)
        self.image = self.image.convert('L')

    def get_info(self):
        """
        Get Image size and mode information using the Pillow library.
        Return:
            (size, mode)
        """
        im = self.image
        print(f"Image size: {im.size}")
        print(f"Image mode: {im.mode}")
        return im.size, im.mode

    def get_array(self):
        """
        Get Numpy array for the input image
        """
        self.array = np.array(self.image, dtype='int64')
        return self.array

    def resize_image(self, width, height):
        new_dim = (width, height)
        self.image = self.image.resize(new_dim)
        return 

    def slice_image(self, xi, yi, size):
        self.slice = self.array[xi:(xi + size), yi:(yi + size)]
        return self.slice

    def show_image(self):
        self.image.show()
        return


def pre_process(image_path, height, width):

    """
    Pre-process an image to produce a resized, grayscale numpy array.
    :param image_path: path to input image
    :param height: resize height
    :param width: resize width
    :return: numpy array with resized dimensions
    """

    im = Process(image_path)
    im.resize_image(width, height)
    array = im.get_array()

    print(f"Pre-processing for {image_path} complete.")
    print("Information:")
    im.get_info()

    return array


def calc_disparity(array_left, array_right, window_size, search_range):

    """
    Calculate a disparity matrix using two image arrays. 
    :param array_left: numpy array of the left image
    :param array_right: numpy array of the right image
    :param window_size: square window size for region matching
    :param search_range: range to apply region matching, in the (-) direction 
    
    :return: disparity matrix numpy array with (dimensions) - (window size)
    """

    start_time = time.time()

    disp_matrix = []

    # print(len(array_left))
    right_border = array_left.shape[0] - window_size

    for row in range(len(array_left)):

        if row % 10 == 0:
            print(f"Disparity calculated for {row} rows.")

        disps = []

        for col1 in range(right_border):
            # print('-------------')
            # print(array_left[row, col1:col1 + window_size])
            # print('-------------')
            win1 = array_left[row, col1:col1 + window_size]
            # print(win1.shape)

            # if col1 < search_range:
            #     init = 0
            # else:
            #     init = col1 - search_range

            if col1 + search_range < right_border:
                search = search_range
            else:
                search = right_border - col1
            # print('diff: ', right_border - col1, 'search: ', search)

            sads = []

            # for col2 in range(col1, init - 1, -1):
            for col2 in range(col1, col1 + search):
                win2 = array_right[row, col2:col2 + window_size]

                sad = np.sum(np.abs(np.subtract(win1, win2)))
                sads.append(sad)

            # print('sads: ', sads)

            disparity = np.argmin(sads)

            # print('disparity: ', disparity)
            disps.append(disparity)

        disp_matrix.append(disps)
                   
    disp_matrix = np.array(disp_matrix)

    end_time = time.time()

    print("Disparity calculations complete.")
    print(f"Time elapsed during disparity calculations: {end_time - start_time}s")

    return disp_matrix

def post_process(disp_matrix):

    pp_disp = np.copy(disp_matrix)

    for x in range(pp_disp.shape[1]):
        for y in range(pp_disp.shape[0]):

            # MEAN
            avg = np.mean(pp_disp[y-7:y+8, x-7:x+8]) 
            if np.absolute(pp_disp[y, x] - avg) > 5:
                pp_disp[y, x] = avg

            # MODE
            if x > 12 and x < (pp_disp.shape[0] - 12):
                if pp_disp[y, x] > 25:
                    mode = stats.mode(pp_disp[y-12:y+13, x-12:x+13].flatten())
                    pp_disp[y, x] = mode[0][0]

            # THRESHOLD
            if pp_disp[y, x] > 30:
                pp_disp[y, x] = 25

    print(f"Post-processing for disparity matrix of shape {disp_matrix.shape} complete.")

    return pp_disp

def create_txt(raw_image_path, disp_matrix, output_path):

    height = disp_matrix.shape[0]
    width = disp_matrix.shape[1]

    img = Image.open(raw_image_path)
    img = img.resize((width, height))
    arr = np.array(img)

    xyzrgb = []

    for x in range(width):
        for y in range(height):
            z = np.multiply(disp_matrix[y, x], 6)
            rgb = arr[y, x]
            xyzrgb.append([x, y, z, rgb[0], rgb[1], rgb[2]])

    df = pd.DataFrame(xyzrgb)
    df.columns = ['x', 'y', 'z', 'r', 'g', 'b']
    df.to_csv(output_path, index=False)

    print(f"Successfully created file at {output_path}.")

    return df

def main():

    variant = 'camera'
    image_path_left  = f'./images/{variant}_left.png'
    image_path_right = f'./images/{variant}_right.png'

    height = 200
    width = 200

    window_size = 5
    search_range = 50
    postproc = False

    array_left = pre_process(image_path=image_path_left,
                             height=height,
                             width=width)
    array_right = pre_process(image_path=image_path_right,
                              height=height,
                              width=width)

    disp_matrix = calc_disparity(array_left=array_left,
                                 array_right=array_right,
                                 window_size=window_size,
                                 search_range=search_range)

    if postproc:
        disp_matrix = post_process(disp_matrix)
        disp_matrix = post_process(disp_matrix)
        disp_matrix = post_process(disp_matrix)
        disp_matrix = post_process(disp_matrix)
        disp_matrix = post_process(disp_matrix)

    # output_txt_path = f"./images/point_cloud_{height}.txt"
    # df = create_txt(raw_image_path=image_path_left,
    #                 disp_matrix=proc_disp_matrix,
    #                 output_path=output_txt_path)
    print(disp_matrix.shape)
    plt.imshow(disp_matrix)
    plt.savefig(f"./images/{variant}_w{window_size}_s{search_range}_postproc{postproc}_{height}.png")
    plt.show()


if __name__ == "__main__":
    main()
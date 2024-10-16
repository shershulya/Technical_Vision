//                                                              Task: Erosion and Dilation of a binary raster.

#include <iostream>
#include <chrono>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main (void) {
    cv::Mat source = cv::imread("./fingerprint_mid.jpg", cv::IMREAD_GRAYSCALE);

    if (!source.data) {
        std::cout << "Could not open or find the image\n";
        return 0;
    }

    cv::Mat bn_source;
    cv::threshold(source, bn_source, 224, 1, cv::THRESH_BINARY_INV);

    uint8_t* data_ptr = (uint8_t*)bn_source.data;


    int padding = 1;
    std::cout << "Window size: " << padding << std::endl;
    int rows_padded = bn_source.rows + 2 * padding;
    int cols_padded = bn_source.cols + 2 * padding;
    
    uint8_t bn_matrix[rows_padded][cols_padded];
    std::fill(*bn_matrix, *bn_matrix + (rows_padded) * (cols_padded), 0);
    for(int i = 0; i < bn_source.rows; i++) {
        for(int j = 0; j < bn_source.cols; j++) {
            bn_matrix[i + 1][j + 1] = data_ptr[i * bn_source.cols + j];
        }
    }
    
    uint8_t eroded[rows_padded][cols_padded];
    std::fill(*eroded, *eroded + (rows_padded) * (cols_padded), 0);

    uint8_t buf[rows_padded][cols_padded];
    std::fill(*buf, *buf + (rows_padded) * (cols_padded), 0);

    int repeats = 1;
    auto averaged_elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(static_cast<std::chrono::microseconds>(0));
    for (int repeat = 0; repeat < repeats; ++repeat) {

        auto begin = std::chrono::steady_clock::now();
        if (padding == 1) {
            for(int i = padding; i < rows_padded - padding; ++i) {
                uint8_t* line = bn_matrix[i];
                for(int j = padding; j < cols_padded - padding; ++j) {
                    buf[i][j] = line[j  - 1] &
                                line[j     ] &
                                line[j  + 1];
                }
            }

            for(int i = padding; i < rows_padded - padding; ++i) {
                for(int j = padding; j < cols_padded - padding; j += 8) {
                    *reinterpret_cast<uint64_t *>(eroded[i] + j) = *reinterpret_cast<uint64_t *>(buf[i - 1] + j) &
                                                                   *reinterpret_cast<uint64_t *>(buf[i    ] + j) &
                                                                   *reinterpret_cast<uint64_t *>(buf[i + 1] + j);
                }
            }
        } else if (padding == 2) {
            for(int i = padding; i < rows_padded - padding; ++i) {
                uint8_t* line = bn_matrix[i];
                for(int j = padding; j < cols_padded - padding; ++j) {
                    buf[i][j] = line[j  - 2] &
                                line[j  - 1] &
                                line[j     ] &
                                line[j  + 1] &
                                line[j  + 2];
                }
            }
            for(int i = padding; i < rows_padded - padding; ++i) {
                for(int j = padding; j < cols_padded - padding; j += 8) {
                    *reinterpret_cast<uint64_t *>(eroded[i] + j) = *reinterpret_cast<uint64_t *>(buf[i - 2] + j) &
                                                                   *reinterpret_cast<uint64_t *>(buf[i - 1] + j) &
                                                                   *reinterpret_cast<uint64_t *>(buf[i    ] + j) &
                                                                   *reinterpret_cast<uint64_t *>(buf[i + 1] + j) &
                                                                   *reinterpret_cast<uint64_t *>(buf[i + 2] + j);
                }
            }
        } else if (padding == 3) {
            for(int i = padding; i < rows_padded - padding; ++i) {
                uint8_t* line = bn_matrix[i];
                for(int j = padding; j < cols_padded - padding; ++j) {            
                    buf[i][j] = line[j  - 3] &
                                line[j  - 2] &
                                line[j  - 1] &
                                line[j     ] &
                                line[j  + 1] &
                                line[j  + 2] &
                                line[j  + 3];
                }
            }
            for(int i = padding; i < rows_padded - padding; ++i) {
                for(int j = padding; j < cols_padded - padding; j += 8) {
                    *reinterpret_cast<uint64_t *>(eroded[i] + j) = *reinterpret_cast<uint64_t *>(buf[i - 3] + j) &
                                                                   *reinterpret_cast<uint64_t *>(buf[i - 2] + j) &
                                                                   *reinterpret_cast<uint64_t *>(buf[i - 1] + j) &
                                                                   *reinterpret_cast<uint64_t *>(buf[i    ] + j) &
                                                                   *reinterpret_cast<uint64_t *>(buf[i + 1] + j) &
                                                                   *reinterpret_cast<uint64_t *>(buf[i + 2] + j) &
                                                                   *reinterpret_cast<uint64_t *>(buf[i + 3] + j);
                }
            }
        } else {
            std::cout << "This window size does not support." << std::endl;
            return 0;
        }
        auto end = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
        averaged_elapsed_ms += elapsed_ms;
    }
    std::cout << "My avg erosion time: " << averaged_elapsed_ms.count() / repeats << " microseconds\n";
    cv::Mat my_erosion = cv::Mat(rows_padded, cols_padded, CV_8U, &eroded);





    int morph_size = padding;
    cv::Mat element = getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
        cv::Point(morph_size, morph_size));
    cv::Mat cv_erosion, cv_dilation;
    
    cv::setNumThreads(0);
    auto averaged_elapsed_ms_erode_cv = std::chrono::duration_cast<std::chrono::microseconds>(static_cast<std::chrono::microseconds>(0));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        auto begin_erode_cv = std::chrono::steady_clock::now();
        
        cv::erode(bn_source, cv_erosion, element, cv::Point(-1, -1), 1);
        
        auto end_erode_cv = std::chrono::steady_clock::now();
        auto elapsed_ms_erode_cv = std::chrono::duration_cast<std::chrono::microseconds>(end_erode_cv - begin_erode_cv);
        averaged_elapsed_ms_erode_cv += elapsed_ms_erode_cv;
    }
    std::cout << "The opencv avg erosion time: " << averaged_elapsed_ms_erode_cv.count() / repeats << " microseconds\n";    



    cv::namedWindow("bn_source", cv::WINDOW_NORMAL);
    cv::resizeWindow("bn_source", 776, 1165);
    cv::moveWindow("bn_source", 50, 80);
    cv::imshow("bn_source", 255 * bn_source);
    cv::namedWindow("cv_erosion", cv::WINDOW_NORMAL);
    cv::resizeWindow("cv_erosion", 776, 1165);
    cv::moveWindow("cv_erosion", 900, 80);
    cv::imshow("cv_erosion", 255 * cv_erosion);
    cv::namedWindow("my_erosion", cv::WINDOW_NORMAL);
    cv::resizeWindow("my_erosion", 776, 1165);
    cv::moveWindow("my_erosion", 1750, 80);
    cv::imshow("my_erosion", 255 * my_erosion);
    cv::waitKey();
    return 0;
}
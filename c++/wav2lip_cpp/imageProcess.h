//
// Created by 张宇 on 2025/3/26.
//

#ifndef WAV2LIP_CPP_IMAGEPROCESS_H
#define WAV2LIP_CPP_IMAGEPROCESS_H

#include "wavDrive.hpp"
#include <opencv2/opencv.hpp>
#include <unsupported/Eigen/CXX11/Tensor>


namespace ImageProcess {
    static WavDrive wavDrive("/Users/zhangyu/CLionProjects/wav2lip_cpp/onnx_models/wav_drive.onnx");

    cv::Mat mergeChannels(cv::Mat &mat1, cv::Mat &mat2) {
        cv::Mat merged(96, 96, CV_8UC(6));

        for (int i = 0; i < mat1.rows; ++i) {
            const uchar *p1 = mat1.ptr<uchar>(i);
            const uchar *p2 = mat2.ptr<uchar>(i);
            auto p_merged = merged.ptr<uchar>(i);

            for (int j = 0; j < mat1.cols; ++j) {
                p_merged[0] = p1[0];
                p_merged[1] = p1[1];
                p_merged[2] = p1[2];
                p_merged[3] = p2[0];
                p_merged[4] = p2[1];
                p_merged[5] = p2[2];

                p1 += 3;
                p2 += 3;
                p_merged += 6;
            }
        }
        return merged;
    }

    void main(
            std::vector<std::vector<std::vector<float>>> &audio_mels_chunks,
            const std::string &target_image_path, std::vector<int> &face_pos) {
        int wav2lip_batch_size = 128;
        auto face_w = face_pos[2] - face_pos[0];
        auto face_h = face_pos[3] - face_pos[1];
        auto target_image = cv::imread(target_image_path);
        auto face = target_image(cv::Rect(face_pos[0], face_pos[1], face_w, face_h));
        cv::resize(face, face, cv::Size(96, 96));
        auto mel_step = static_cast<int>(std::ceil(static_cast<double>(audio_mels_chunks.size()) / wav2lip_batch_size));
        std::vector<std::vector<std::vector<float>>> mel_batch;
        std::vector<cv::Mat> image_batch, faces_masked;
        for (size_t i = 0; i < mel_step; ++i) {
            image_batch.clear();
            if (i == mel_step - 1) {
                mel_batch.assign(audio_mels_chunks.begin() + i * wav2lip_batch_size,
                                 (audio_mels_chunks.begin() + i * wav2lip_batch_size +
                                  (audio_mels_chunks.size() - i * wav2lip_batch_size)));
                for (size_t _ = 0; _ < audio_mels_chunks.size() - i * wav2lip_batch_size; ++_) {
                    image_batch.emplace_back(face);
                }
            } else {
                mel_batch.assign(audio_mels_chunks.begin() + i * wav2lip_batch_size,
                                 audio_mels_chunks.begin() + (i + 1) * wav2lip_batch_size);
                for (size_t _ = 0; _ < wav2lip_batch_size; ++_) {
                    image_batch.emplace_back(face);
                }
            }
            for (auto &face: image_batch) {
                cv::Mat face_masked;
                face.copyTo(face_masked);
                for (int y = 0; y < face_masked.rows; y++) {
                    for (int x = (96 / 2); x < 96; x++) {
                        auto ptr = face_masked.ptr<uchar>(x, y);
                        for (int c = 0; c < face_masked.channels(); c++) {
                            ptr[c] = 0;
                        }
                    }
                }
                faces_masked.emplace_back(face_masked);
            }
            std::vector<cv::Mat> merge_batch;
            for (size_t i = 0; i < wav2lip_batch_size; ++i) {
                auto face_masked = faces_masked[i];
                auto image = image_batch[i];
                auto merge_image = ImageProcess::mergeChannels(face_masked, image);
                merge_image.convertTo(merge_image, CV_32F);
                merge_batch.emplace_back(merge_image);
            }

            Eigen::Tensor<float, 4> batch_mel_tensors(128, 1, (int) mel_batch[0].size(), (int) mel_batch[0][0].size());
            for (size_t i = 0; i < mel_batch.size(); ++i) {
                for (size_t j = 0; j < mel_batch[0].size(); ++j) {
                    for (size_t k = 0; k < mel_batch[0][0].size(); ++k) {
                        batch_mel_tensors((int) i, 0, (int) j, (int) k) = mel_batch[i][j][k];
                    }
                }
            }
            Eigen::Tensor<float, 4> batch_image_tensors(128, 6, 96, 96);
            for (size_t i = 0; i < merge_batch.size(); ++i) {
                for (size_t c = 0; c < 6; ++c) {
                    for (size_t h = 0; h < 96; ++h) {
                        for (size_t w = 0; w < 96; ++w) {
                            // 注意：OpenCV中的数据是行优先的，而Eigen::Tensor默认也是行优先
                            batch_image_tensors((int) i, (int) c, (int) h, (int) w) = merge_batch[i].at<cv::Vec6i>(h,
                                                                                                                   w)[c];
                        }
                    }
                }
            }
            std::cout << batch_mel_tensors.dimensions() << std::endl;
            std::cout << batch_image_tensors.dimensions() << std::endl;
            wavDrive.call(batch_mel_tensors, batch_image_tensors);
        }
    }
}

#endif //WAV2LIP_CPP_IMAGEPROCESS_H

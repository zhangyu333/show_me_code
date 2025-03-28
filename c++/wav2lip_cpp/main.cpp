#include <iostream>
#include "lmsDet.hpp"
#include "audioProcess.h"
#include "imageProcess.h"

int main() {
    const std::string wav_path = "/Users/zhangyu/CLionProjects/wav2lip_cpp/test.wav";
    const std::string lms_model_path = "/Users/zhangyu/CLionProjects/wav2lip_cpp/onnx_models/lms.onnx";
    const std::string target_image_path = "/Users/zhangyu/CLionProjects/wav2lip_cpp/test.png";
    DetLMS lmsDetector(lms_model_path);
    std::vector<int> face_pos = {68, 129, 330, 406};
    auto audio_mels_chunks = AudioProcess::main(wav_path);
    ImageProcess::main(audio_mels_chunks, target_image_path, face_pos);
    return 0;
}

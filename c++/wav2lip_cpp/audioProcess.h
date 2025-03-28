//
// Created by 张宇 on 2025/3/25.
//

#ifndef WAV2LIP_CPP_AUDIOPROCESS_H
#define WAV2LIP_CPP_AUDIOPROCESS_H

#include "model.h"
#include "AudioFile.h"
#include "usePy.hpp"

namespace AudioProcess {
    static Model::AudioParams audio_params{
            16000, 800, 200, "hann",
            false, "reflect", 2.f, 80, 55,
            7600, 20, true, 2};

    AudioFile<double> loadAudio(const std::string &audio_path) {
        AudioFile<double> audio_file;
        audio_file.load(audio_path);
        audio_file.printSummary();
        return audio_file;
    }

    std::vector<std::vector<std::vector<float>>> audioProc(AudioFile<double> &audio_file) {
        int channel = 0;
        auto wav = audio_file.samples[channel];
        auto x_features = callPythonFunction(wav);
        return x_features;
    };

    std::vector<std::vector<std::vector<float>>>
    main(const std::string &audio_path) {
        auto audio = AudioProcess::loadAudio(audio_path);
        auto audio_proc = AudioProcess::audioProc(audio);
        return audio_proc;
    };
}

#endif //WAV2LIP_CPP_AUDIOPROCESS_H

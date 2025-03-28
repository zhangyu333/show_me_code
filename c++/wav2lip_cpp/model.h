//
// Created by 张宇 on 2025/3/25.
//

#ifndef WAV2LIP_CPP_MODEL_H
#define WAV2LIP_CPP_MODEL_H

#include <iostream>

namespace Model {

    typedef struct {
        int sr;  // audio sample_rate
        int n_fft; // length of the FFT size
        int n_hop; //number of samples between successive frames
        std::string window; //window function. currently only supports 'hann'
        bool center; //same as librosa
        std::string pad_mode; // pad mode. support "reflect","symmetric","edge"
        float power; //exponent for the magnitude melspectrogram
        int n_mel; //number of mel bands
        int f_min; //lowest frequency (in Hz)
        int f_max; // highest frequency (in Hz)
        int n_mfcc; //number of mfccs
        bool norm; //ortho-normal dct basis
        int type; //dct type. currently only supports 'type-II'
    } AudioParams;

}


#endif //WAV2LIP_CPP_MODEL_H

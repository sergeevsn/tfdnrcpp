#ifndef TFD_H
#define TFD_H

#include <vector>
#include <complex>
#include <string>
#include <stdexcept>
#include <cstddef>

namespace tfd {

using complex = std::complex<float>;
using real_vector = std::vector<float>;
using complex_vector = std::vector<complex>;
using seismic_data = std::vector<real_vector>;
using complex_spectrogram = std::vector<complex_vector>;

real_vector window_hann(size_t n);

real_vector reflect_padding(const real_vector& trace, size_t target_length);

float compute_median(real_vector values);

complex_spectrogram compute_stft(const seismic_data& seismic, size_t n_fft, size_t n_overlap);

seismic_data compute_istft(const complex_spectrogram& spectrograms, size_t n_fft, size_t n_overlap, size_t original_length);

complex_spectrogram tfd_noise_rejection(const complex_spectrogram& stft_data, size_t trace_aperture,
                                        float threshold_multiplier, size_t n_fft,
                                        bool use_global_threshold = true);
}

#endif // TFD_H

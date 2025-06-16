#include "tfd.h"
#include <fftw3.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace tfd {

// Optimized window calculation
real_vector window_hann(size_t n) {
    real_vector window(n);
    if (n == 0) return window;
    if (n == 1) {
        window[0] = 1.0f;
        return window;
    }
    const float factor = 2.0f * M_PI / (n - 1);
    for (size_t i = 0; i < n; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(factor * i));
    }
    return window;
}

// Vectorized reflection padding
real_vector reflect_padding(const real_vector& trace, size_t target_length) {
    if (trace.size() >= target_length) {
        return trace;
    }
    real_vector padded;
    padded.reserve(target_length);
    padded.insert(padded.end(), trace.begin(), trace.end());

    // Create reflection part once
    real_vector reflection(trace.rbegin() + 1, trace.rend());
    if (reflection.empty()) { // Handle single-point traces
        if (!trace.empty()) {
            padded.resize(target_length, trace.front());
        }
        return padded;
    }
    
    // Append reflected part as many times as needed
    while (padded.size() < target_length) {
        size_t remaining = target_length - padded.size();
        padded.insert(padded.end(), reflection.begin(), reflection.begin() + std::min(reflection.size(), remaining));
    }
    return padded;
}

//Optimized median calculation using nth_element
float compute_median(real_vector values) {
    if (values.empty()) return 0.0f;
    size_t n = values.size();
    auto mid_it = values.begin() + n / 2;
    std::nth_element(values.begin(), mid_it, values.end());
    float median = *mid_it;

    if (n % 2 == 0) {
        // For even size, median is the average of the two middle elements.
        // We found the upper-middle element. Now find the max of the lower-half,
        // which is the lower-middle element after partitioning.
        auto lower_mid_it = std::max_element(values.begin(), mid_it);
        median = (*lower_mid_it + median) / 2.0f;
    }
    return median;
}

complex_spectrogram compute_stft(const seismic_data& seismic, size_t n_fft, size_t n_overlap) {
    if (n_fft == 0 || n_overlap >= n_fft)
        throw std::invalid_argument("Invalid STFT parameters");

    complex_spectrogram spectrograms;
    const size_t window_step = n_fft - n_overlap;
    const real_vector window = window_hann(n_fft);

    std::vector<float> in(n_fft);
    std::vector<fftwf_complex> out(n_fft/2 + 1);
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(n_fft, in.data(), out.data(), FFTW_MEASURE);

    for (const auto& trace : seismic) {
        if (trace.empty()) continue;

        // Correct calculation for number of frames needed to cover the original trace
        size_t n_frames = (trace.size() > n_fft) ? 1 + (trace.size() - n_fft) / window_step : 1;
        size_t padded_length = (n_frames - 1) * window_step + n_fft;
        const real_vector padded = reflect_padding(trace, padded_length);

        complex_vector spectrum(n_frames * (n_fft / 2 + 1));
        for (size_t frame = 0; frame < n_frames; ++frame) {
            size_t offset = frame * window_step;
            
            std::transform(padded.begin() + offset, 
                           padded.begin() + offset + n_fft,
                           window.begin(),
                           in.begin(),
                           std::multiplies<>());

            fftwf_execute(plan);

            size_t spectrum_offset = frame * (n_fft / 2 + 1);
            for (size_t i = 0; i < n_fft / 2 + 1; ++i) {
                spectrum[spectrum_offset + i] = {out[i][0], out[i][1]};
            }
        }
        spectrograms.emplace_back(std::move(spectrum));
    }

    fftwf_destroy_plan(plan);
    return spectrograms;
}

seismic_data compute_istft(const complex_spectrogram& spectrograms, 
                           size_t n_fft, size_t n_overlap, 
                           size_t original_length) 
{
    seismic_data reconstructed;
    if (spectrograms.empty() || n_fft == 0 || n_overlap >= n_fft) return reconstructed;
    
    const size_t window_step = n_fft - n_overlap;
    const real_vector window = window_hann(n_fft);
    const real_vector window_sq = [&]{
        real_vector sq(n_fft);
        std::transform(window.begin(), window.end(), sq.begin(),
                       [](float w){ return w * w; });
        return sq;
    }(); 

    std::vector<fftwf_complex> in(n_fft / 2 + 1);
    std::vector<float> out(n_fft);
    fftwf_plan plan = fftwf_plan_dft_c2r_1d(n_fft, in.data(), out.data(), FFTW_MEASURE);

    for (const auto& spectrum : spectrograms) {
        if (spectrum.empty()) continue;
        const size_t n_freq = n_fft / 2 + 1;
        const size_t n_frames = spectrum.size() / n_freq;
        const size_t output_size = (n_frames - 1) * window_step + n_fft;
        
        real_vector output(output_size, 0.0f);
        real_vector overlap_count(output_size, 0.0f);

        for (size_t frame = 0; frame < n_frames; ++frame) {
            size_t offset = frame * window_step;
            size_t spectrum_offset = frame * n_freq;

            for (size_t i = 0; i < n_freq; ++i) {
                in[i][0] = spectrum[spectrum_offset + i].real();
                in[i][1] = spectrum[spectrum_offset + i].imag();
            }

            fftwf_execute(plan);
            
            for (size_t i = 0; i < n_fft; ++i) {
                if (offset + i >= output_size) break;
                output[offset + i] += out[i] * window[i] / n_fft;                
                overlap_count[offset + i] += window_sq[i];
            }
        }
        
        // Normalize by overlap-add window sum
        std::transform(output.begin(), output.end(), overlap_count.begin(), output.begin(),
                       [](float val, float cnt) { return cnt > 1e-8f ? val / cnt : 0.0f; });
      
        // Trim to original length
        if (output.size() > original_length) {
            output.resize(original_length);
        }

        reconstructed.emplace_back(std::move(output));
    }
    
    fftwf_destroy_plan(plan);
    return reconstructed;
}


complex_spectrogram tfd_noise_rejection(
    const complex_spectrogram& stft_data,
    size_t trace_aperture,
    float threshold_multiplier,
    size_t n_fft,
    bool use_global_threshold)
{
    complex_spectrogram filtered = stft_data;
    const size_t n_traces = stft_data.size();
    if (n_traces < 2 * trace_aperture + 1) {
        // Not enough traces to apply the filter meaningfully, return original
        return filtered;
    }

    const size_t n_freq = n_fft / 2 + 1;
    if (stft_data[0].empty() || stft_data[0].size() % n_freq != 0) {
        throw std::runtime_error("Spectrogram dimensions do not match n_fft.");
    }
    const size_t n_frames = stft_data[0].size() / n_freq;

    // --- Threshold Calculation ---
    std::vector<float> thresholds(n_freq);
    if (use_global_threshold) {
        real_vector all_amplitudes;
        all_amplitudes.reserve(n_traces * n_frames * n_freq);
        for (const auto& trace_spectrum : stft_data) {
            for (const auto& c : trace_spectrum) {
                all_amplitudes.push_back(std::abs(c));
            }
        }
        float global_threshold = compute_median(all_amplitudes) * threshold_multiplier;
        std::fill(thresholds.begin(), thresholds.end(), global_threshold);
    } else {
        // Frequency-dependent thresholding
        for (size_t f = 0; f < n_freq; ++f) {
            real_vector amps_for_this_freq;
            amps_for_this_freq.reserve(n_traces * n_frames);
            for (size_t i = 0; i < n_traces; ++i) {
                for (size_t t = 0; t < n_frames; ++t) {
                    amps_for_this_freq.push_back(std::abs(stft_data[i][t * n_freq + f]));
                }
            }
            thresholds[f] = compute_median(amps_for_this_freq) * threshold_multiplier;
        }
    }

    // Pre-calculate phases to avoid repeated calls to std::arg
    std::vector<real_vector> phases(n_traces);
    for (size_t i = 0; i < n_traces; ++i) {
        phases[i].resize(stft_data[i].size());
        std::transform(stft_data[i].begin(), stft_data[i].end(), phases[i].begin(),
                       [](const complex& c) { return std::arg(c); });
    }

    // --- Main Filtering Loop ---
    for (size_t i = 0; i < n_traces; ++i) {
        for (size_t t = 0; t < n_frames; ++t) {
            for (size_t f = 0; f < n_freq; ++f) {
                size_t idx = t * n_freq + f;
                float original_amp = std::abs(stft_data[i][idx]);

                // Apply filter only if amplitude is above the determined threshold
                if (original_amp > thresholds[f]) {
                    
                    // Define the spatial window for the median filter
                    size_t start_trace = (i > trace_aperture) ? i - trace_aperture : 0;
                    size_t end_trace = std::min(i + trace_aperture + 1, n_traces);
                    
                    // Collect amplitudes from neighboring traces at the same T-F point
                    real_vector spatial_window;
                    spatial_window.reserve(end_trace - start_trace);
                    for (size_t j = start_trace; j < end_trace; ++j) {
                        spatial_window.push_back(std::abs(stft_data[j][idx]));
                    }
                    
                    // Replace amplitude with the spatial median, keep original phase
                    float median_amp = compute_median(spatial_window);
                    filtered[i][idx] = std::polar(median_amp, phases[i][idx]);
                }
            }
        }
    }

    return filtered;
}

} // namespace tfd

## TFDNR.CPP

C++ STFT Median Thresholding Filter, based on the FreeUSP TFDNoise utility. The filtering function itself is located in ```tfd.cpp```. The utility ``tfdnoise`` takes a SEG-Y file, filters all gathers within it, and stores the result in a new SEG-Y file, while also creating a difference SEG-Y file. Parameters for the filtering process are stored in ```tfdparams.py```. The utility ```makepics.py``` generates a comparison plot of these three SEG-Y files for specified trace numbers.

First, download, build and install ```FFTW3``` for spectrum calcuation: https://github.com/FFTW/fftw3.git

### Usage
```C++

#include "tfd.h"

// Parameters for tfd::tfd_noise_rejection
const size_t n_fft = 32; 
const size_t n_overlap = 16; // usually 1/2 of n_fft
const size_t trace_aperture = 15;
const float threshold_multiplier = 10.0f;
const bool global_median = true; // variant with one median for all frequencies

tfd::seismic_data seismogram; 

auto stft = tfd::compute_stft(seismogram, n_fft, n_overlap);            
auto filtered_stft = tfd::tfd_noise_rejection(stft, trace_aperture, threshold_multiplier, n_fft, global_median);
auto filtered_seismogram = tfd::compute_istft(filtered_stft, n_fft, n_overlap, N); 
```

### Compile and run test program
```bash
./compile.sh
./tfdnr test.sgy test_processed.sgy
```

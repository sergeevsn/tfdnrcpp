#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <string>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <map> 
#include <fftw3.h>
#include "tfd.h"

// constats for IBM to IEEE conversion
#define SEGYIO_IEMAXIB 0x7fffffff 
#define SEGYIO_IEEEMAX 0x7f7fffff 
#define SEGYIO_IEMINIB 0x00ffffff 

// Parameters for tfd::tfd_noise_rejection
const size_t n_fft = 32; 
const size_t n_overlap = 16; // usually 1/2 of n_fft
const size_t trace_aperture = 15;
const float threshold_multiplier = 10.0f;
const bool global_median = true; // variant with one median for all frequencies

// --- Helper Functions for Data Conversion ---

uint16_t swap_bytes_16(uint16_t val) {
    return (val << 8) | (val >> 8);
}

uint32_t swap_bytes_32(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

float ibm_to_ieee(uint32_t ibm) {    
    if (ibm == 0) return 0.0f;
  
    static const int it[8] = { 0x21800000, 0x21400000, 0x21000000, 0x21000000,
                               0x20c00000, 0x20c00000, 0x20c00000, 0x20c00000 };
    static const int mt[8] = { 8, 4, 2, 2, 1, 1, 1, 1 };
   
    uint32_t manthi = ibm & 0x00ffffff;    
    
    int ix = manthi >> 21;    
    
    uint32_t iexp = ((ibm & 0x7f000000) - it[ix]) << 1;    
    
    manthi = manthi * mt[ix] + iexp;
    
    
    uint32_t inabs = ibm & 0x7fffffff;
    if (inabs > SEGYIO_IEMAXIB) manthi = SEGYIO_IEEEMAX;
  
    manthi = manthi | (ibm & 0x80000000);
    
    uint32_t result_bits = (inabs < SEGYIO_IEMINIB) ? 0 : manthi;
    
    float result_float;
    std::memcpy(&result_float, &result_bits, sizeof(float));
    return result_float;
}
uint32_t ieee_to_ibm(float f) {
    
    uint32_t ieee;
    std::memcpy(&ieee, &f, sizeof(uint32_t));
    
    if (ieee == 0) return 0;
    
    static const int it[4] = { 0x21200000, 0x21400000, 0x21800000, 0x22100000 };
    static const int mt[4] = { 2, 4, 8, 1 };
    
    int ix = (ieee & 0x01800000) >> 23;
    
    uint32_t iexp = ((ieee & 0x7e000000) >> 1) + it[ix];
   
    uint32_t manthi = (mt[ix] * (ieee & 0x007fffff)) >> 3;
    
    uint32_t ibm_bits = (manthi + iexp) | (ieee & 0x80000000);
    
    return (ieee & 0x7fffffff) ? ibm_bits : 0;
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_segy_file> <output_segy_file>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    
    std::cout << "Seismic gather TFD Noise Rejection: C++ implementation" << std::endl;
    if (global_median) {
        std::cout << "Threshold calculation: common median for all frequency bins" << std::endl;
    }
    else {
        std::cout << "Threshold calculation: separate medians for every frequency bin" << std::endl;
    }

    try {
        // Copy input file to output
        std::ifstream src(input_path, std::ios::binary);
        if (!src) {
            throw std::runtime_error("Cannot open input file for copying: " + input_path);
        }
        std::ofstream dst(output_path, std::ios::binary);
        if (!dst) {
            throw std::runtime_error("Cannot open output file for copying: " + output_path);
        }
        dst << src.rdbuf();
        src.close();
        dst.close();
       
        std::ifstream fin(input_path, std::ios::binary);
        std::fstream fout(output_path, std::ios::in | std::ios::out | std::ios::binary);
        if (!fin || !fout) {
            throw std::runtime_error("Error opening files for processing.");
        }
       
        fin.seekg(3216);
        uint16_t dt_us;
        fin.read(reinterpret_cast<char*>(&dt_us), sizeof(dt_us));
        dt_us = swap_bytes_16(dt_us);       
       
        if (dt_us == 0) {
            throw std::runtime_error("Sample interval (dt) is zero in binary header.");
        }        
        

        fin.seekg(3220);
        uint16_t n_samples_per_trace;
        fin.read(reinterpret_cast<char*>(&n_samples_per_trace), sizeof(n_samples_per_trace));
        n_samples_per_trace = swap_bytes_16(n_samples_per_trace);
        const size_t N = n_samples_per_trace;

        if (N == 0) {
            throw std::runtime_error("Number of samples per trace is zero in binary header.");
        }       

        const size_t trace_header_size = 240;
        const size_t trace_data_size = N * sizeof(uint32_t);
        const size_t full_trace_size = trace_header_size + trace_data_size;             
       

        std::map<int32_t, std::vector<std::streampos>> groups;
        fin.seekg(3600);

        char header_buffer[trace_header_size];   
       
        
        while (fin.read(header_buffer, trace_header_size)) {
            std::streampos current_pos = fin.tellg();
            current_pos -= trace_header_size; 

            int32_t field_record_number;            
            memcpy(&field_record_number, header_buffer + 8, sizeof(int32_t));
            field_record_number = static_cast<int32_t>(swap_bytes_32(field_record_number));            
            groups[field_record_number].push_back(current_pos);
            
            fin.seekg(trace_data_size, std::ios::cur);
        }
        
        fin.clear();     
        
        std::cout << "DEBUG: n_samples_per_trace = " << n_samples_per_trace << std::endl;
        std::cout << "DEBUG: dt_us = " << dt_us << std::endl;
        std::cout << "DEBUG: Number of gathers aka groups.size() = " << groups.size() << std::endl;
        
        
         // Start calculation 
        std::cout << "Starting TFD Noise Rejection..." << std::endl; 
        auto start_time = std::chrono::high_resolution_clock::now();        
       
        std::vector<char> trace_buffer(full_trace_size);

        for (const auto& [record_num, positions] : groups) {
            tfd::seismic_data seismic_group;            
            
            for (const auto& pos : positions) {
                fin.seekg(pos);
                fin.read(trace_buffer.data(), full_trace_size);

                tfd::real_vector trace(N);
                char* data_start = trace_buffer.data() + trace_header_size;
                for (size_t i = 0; i < N; ++i) {
                    uint32_t ibm;
                    memcpy(&ibm, data_start + i * sizeof(uint32_t), sizeof(ibm));
                    ibm = swap_bytes_32(ibm);
                    trace[i] = ibm_to_ieee(ibm);
                }
                seismic_group.push_back(std::move(trace));
            }
            
            auto stft = tfd::compute_stft(seismic_group, n_fft, n_overlap);            
            auto filtered_stft = tfd::tfd_noise_rejection(stft, trace_aperture, threshold_multiplier, n_fft, global_median);
            auto recon_group = tfd::compute_istft(filtered_stft, n_fft, n_overlap, N);                        
          
            for (size_t i = 0; i < positions.size(); ++i) {                
                fin.seekg(positions[i]);
                fin.read(trace_buffer.data(), full_trace_size);                
            
                const auto& trace_out = recon_group[i];
                char* data_start = trace_buffer.data() + trace_header_size;
                for (size_t j = 0; j < N; ++j) {
                    float sample = (j < trace_out.size()) ? trace_out[j] : 0.0f;
                
                    uint32_t ibm = ieee_to_ibm(sample);
                    ibm = swap_bytes_32(ibm); 
                    memcpy(data_start + j * sizeof(uint32_t), &ibm, sizeof(ibm));
                }                
                fout.seekp(positions[i]);
                fout.write(trace_buffer.data(), full_trace_size);
            }
        }

        fin.close();
        fout.close();

        std::cout << "Processing complete. Output saved to " << output_path << std::endl;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
 
        std::cout << "\nTotal execution time: " 
              << std::fixed << std::setprecision(3) << elapsed.count() 
              << " seconds\n";


    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

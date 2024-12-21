// added by i.jeong
// code for load CIFAR10 data

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_CIFAR10_LOADER_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_CIFAR10_LOADER_H_

#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

class CIFAR10Loader {
private:
    std::vector<uint8_t> dataset_buffer;  
    std::vector<uint8_t> labels;          
    size_t num_images = 0;                

public:
    static const int IMAGE_SIZE = 32;
    static const int NUM_CHANNELS = 3;
    static const int LABEL_BYTES = 1;
    static const int BYTES_PER_IMAGE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + LABEL_BYTES;
    static const int IMAGES_PER_FILE = 10000;

    struct Image {
        const uint8_t* data;  
        uint8_t label;        
    };

    // load binary file
    bool loadFile(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // check file size
        file.seekg(0, std::ios::end);
        size_t filesize = file.tellg();
        file.seekg(0, std::ios::beg);

        if (filesize != IMAGES_PER_FILE * BYTES_PER_IMAGE) {
            return false;
        }
        size_t current_pos = dataset_buffer.size();
        
        dataset_buffer.resize(current_pos + IMAGES_PER_FILE * IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS);
        labels.resize(labels.size() + IMAGES_PER_FILE);

        for (int i = 0; i < IMAGES_PER_FILE; i++) {
            // read label
            file.read(reinterpret_cast<char*>(&labels[num_images + i]), LABEL_BYTES);
            
            // read image
            file.read(reinterpret_cast<char*>(dataset_buffer.data() + current_pos + i * IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS), 
                     IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS);
        }

        num_images += IMAGES_PER_FILE;
        return true;
    }

    // load multiple files (ex: data_batch_1.bin ~ data_batch_5.bin)
    bool loadFiles(const std::vector<std::string>& filepaths) {
        bool success = true;
        for (const auto& filepath : filepaths) {
            success &= loadFile(filepath);
        }
        return success;
    }

    Image getImage(size_t index) const {
        if (index >= num_images) {
            throw std::out_of_range("Image index out of range");
        }

        Image img;
        img.label = labels[index];
        img.data = dataset_buffer.data() + (index * IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS);
        return img;
    }

    size_t size() const { return num_images; }
    
    static const char* getClassName(uint8_t label) {
        static const char* class_names[] = {
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        };
        return (label < 10) ? class_names[label] : "unknown";
    }
};
#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_CIFAR10_LOADER_H_
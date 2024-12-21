// added by i.jeong
// code for load fmnist data

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_FMNIST_LOADER_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_FMNIST_LOADER_H_


#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>

class FMNISTLoader {
private:
    // Helper function to reverse endianness
    uint32_t reverseInt(uint32_t i) {
        uint8_t c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
    }

public:
    struct FMNISTData {
        std::vector<std::vector<uint8_t>> images;
        std::vector<uint8_t> labels;
        int width;
        int height;
    };

    FMNISTData loadData(const std::string& imageFile, const std::string& labelFile) {
        FMNISTData data;
        
        // Read images
        std::ifstream imageStream(imageFile, std::ios::binary);
        if (!imageStream.is_open()) {
            throw std::runtime_error("Cannot open image file: " + imageFile);
        }

        uint32_t magic;
        uint32_t numImages;
        uint32_t numRows;
        uint32_t numCols;

        imageStream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        imageStream.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
        imageStream.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
        imageStream.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

        // Reverse endianness
        magic = reverseInt(magic);
        numImages = reverseInt(numImages);
        numRows = reverseInt(numRows);
        numCols = reverseInt(numCols);

        if (magic != 2051) {
            throw std::runtime_error("Invalid image file format");
        }

        data.width = numCols;
        data.height = numRows;

        // Read image data
        data.images.resize(numImages, std::vector<uint8_t>(numRows * numCols));
        for (uint32_t i = 0; i < numImages; i++) {
            imageStream.read(reinterpret_cast<char*>(data.images[i].data()), numRows * numCols);
        }

        // Read labels
        std::ifstream labelStream(labelFile, std::ios::binary);
        if (!labelStream.is_open()) {
            throw std::runtime_error("Cannot open label file: " + labelFile);
        }

        labelStream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        labelStream.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));

        magic = reverseInt(magic);
        numImages = reverseInt(numImages);

        if (magic != 2049) {
            throw std::runtime_error("Invalid label file format");
        }

        // Read label data
        data.labels.resize(numImages);
        labelStream.read(reinterpret_cast<char*>(data.labels.data()), numImages);

        return data;
    }
};

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_FMNIST_LOADER_H_
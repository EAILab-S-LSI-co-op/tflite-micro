#include "cifar10_loader.h"
#define data_batch_1_path "/workspace/tflm/tflite-micro/tensorflow/lite/micro/examples/cifar10/data/data_batch_1.bin"


int main() {
    CIFAR10Loader loader;
    
    try {
        // 2. 트레이닝 데이터 로드 (예: data_batch_1.bin 파일)
        auto dataset = loader.loadBinaryFile(data_batch_1_path);
        
        // 3. 첫 번째 이미지 정보 출력
        const auto& first_image = dataset.images[0];
        std::cout << "First image label: " << 
            loader.getClassName(first_image.label) << std::endl;
        
        // 4. 첫 번째 픽셀의 RGB 값 출력
        uint8_t r, g, b;
        loader.getRGB(first_image, 0, 0, r, g, b);
        std::cout << "First pixel RGB: (" << 
            (int)r << "," << (int)g << "," << (int)b << ")" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/cifar10/models/generated_cifar10_float_model.h"
#include "tensorflow/lite/micro/examples/cifar10/models/generated_cifar10_int8_model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// added by i.jeong
// code for load cifar10 data
#define DATA_IDX 0
#define NUM_CLASS 10
#define NUM_SAMPLES 10000
#define test_data_path "/workspace/tflm/tflite-micro/tensorflow/lite/micro/examples/cifar10/data/test_batch.bin"
#include "tensorflow/lite/micro/examples/cifar10/cifar10_loader.h"

CIFAR10Loader loader;
auto data = loader.loadFile(test_data_path);

namespace {

// added by i.jeong
// change op_resolver size 
using OpResolver = tflite::MicroMutableOpResolver<5>;

TfLiteStatus RegisterOps(OpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaxPool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus ProfileMemoryAndLatency() {
  tflite::MicroProfiler profiler;
  OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 224224;
  uint8_t tensor_arena[kTensorArenaSize];
  constexpr int kNumResourceVariables = 24;

  tflite::RecordingMicroAllocator* allocator(
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(cifar10_float_tflite), op_resolver, allocator,
      tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables),
      &profiler);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  TFLITE_CHECK_EQ(interpreter.inputs_size(), 1);
  interpreter.input(0)->data.f[0] = 1.f;
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  MicroPrintf("");  // Print an empty new line
  profiler.LogTicksPerTagCsv();

  MicroPrintf("");  // Print an empty new line
  interpreter.GetMicroAllocator().PrintAllocations();
  return kTfLiteOk;
}

TfLiteStatus LoadFloatModelAndPerformInference() {
  const tflite::Model* model =
      ::tflite::GetModel(cifar10_float_tflite);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 224224;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  // Added by i.jeong
  // setup input and perform inference
  auto img = loader.getImage(DATA_IDX);

  // need to rearrange the input data
  // need to change rrr ggg bbb -> rgb rgb rgb
  uint16_t num_pixels = loader.IMAGE_SIZE * loader.IMAGE_SIZE; 
  for (uint16_t i = 0; i < num_pixels; i++){
    interpreter.input(0)->data.f[i*loader.NUM_CHANNELS + 0] = img.data[i] / 255.0;
    interpreter.input(0)->data.f[i*loader.NUM_CHANNELS + 1] = img.data[num_pixels + i] / 255.0;
    interpreter.input(0)->data.f[i*loader.NUM_CHANNELS + 2] = img.data[num_pixels * 2 + i] / 255.0;
  }
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  uint8_t y_pred = std::distance(interpreter.output(0)->data.f, std::max_element(interpreter.output(0)->data.f, interpreter.output(0)->data.f + NUM_CLASS));
  MicroPrintf("[float model] y_pred: %d", y_pred);
  MicroPrintf("[float model] y_real: %d\n", (int)(img.label));

  return kTfLiteOk;
}

TfLiteStatus LoadFloatModelAndPerformInferenceForAllData() {
  const tflite::Model* model =
      ::tflite::GetModel(cifar10_float_tflite);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 224224;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  // Added by i.jeong
  // setup input and perform inference
  double accuracy = 0.0;
  uint16_t num_pixels = loader.IMAGE_SIZE * loader.IMAGE_SIZE; 

  for (uint16_t data_idx = 0; data_idx < loader.size(); data_idx++){
    auto img = loader.getImage(data_idx);

    // need to rearrange the input data
    // need to change rrr ggg bbb -> rgb rgb rgb
    for (uint16_t i = 0; i < num_pixels; i++){
      interpreter.input(0)->data.f[i*loader.NUM_CHANNELS + 0] = img.data[i] / 255.0;
      interpreter.input(0)->data.f[i*loader.NUM_CHANNELS + 1] = img.data[num_pixels + i] / 255.0;
      interpreter.input(0)->data.f[i*loader.NUM_CHANNELS + 2] = img.data[num_pixels * 2 + i] / 255.0;
    }
    TF_LITE_ENSURE_STATUS(interpreter.Invoke());

    uint8_t y_pred = std::distance(interpreter.output(0)->data.f, std::max_element(interpreter.output(0)->data.f, interpreter.output(0)->data.f + NUM_CLASS));
    if (y_pred == (uint8_t)img.label){
      accuracy += 1.0;
    }
  }
  accuracy /= loader.size();
  MicroPrintf("[float model] accuracy: %f", accuracy);

  return kTfLiteOk;
}

TfLiteStatus LoadQuantModelAndPerformInference() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(cifar10_int8_tflite);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 224224;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  TfLiteTensor* input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);

  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);


  // // added by i.jeong
  // // code for inference of quantized model

  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;
  
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  auto img = loader.getImage(DATA_IDX);

  // need to rearrange the input data
  // need to change rrr ggg bbb -> rgb rgb rgb
  uint16_t num_pixels = loader.IMAGE_SIZE * loader.IMAGE_SIZE;
  for (uint16_t i = 0; i < num_pixels; i++){
    int _quantized = round(img.data[i] / input_scale) + input_zero_point;
    input->data.int8[i*loader.NUM_CHANNELS + 0] = std::max(-128, std::min(127, _quantized));

    _quantized = round(img.data[i + num_pixels] / input_scale) + input_zero_point;
    input->data.int8[i*loader.NUM_CHANNELS + 1] = std::max(-128, std::min(127, _quantized));

    _quantized = round(img.data[i + 2*num_pixels] / input_scale) + input_zero_point;
    input->data.int8[i*loader.NUM_CHANNELS + 2] = std::max(-128, std::min(127, _quantized));
  }
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());


  std::vector<float> dequantized_output(NUM_CLASS);
  for (uint8_t i = 0; i < NUM_CLASS; i++) {
    dequantized_output[i] = (output->data.int8[i] - output_zero_point) * output_scale;
  }
  MicroPrintf("[int8 model] y_pred: %d", std::distance(dequantized_output.begin(), std::max_element(dequantized_output.begin(), dequantized_output.end())));
  MicroPrintf("[int8 model] y_real: %d\n", (int)(img.label));
  return kTfLiteOk;
}

TfLiteStatus LoadQuantModelAndPerformInferenceForAllData() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(cifar10_int8_tflite);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 224224;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  TfLiteTensor* input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);

  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);


  // added by i.jeong
  // code for inference of quantized model
  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;
  
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  double accuracy = 0.0;
  uint16_t num_pixels = loader.IMAGE_SIZE * loader.IMAGE_SIZE;
  for (uint16_t data_idx = 0; data_idx < loader.size(); data_idx++){
    auto img = loader.getImage(data_idx);
    for (uint16_t i = 0; i < num_pixels; i++){
      int _quantized = round(img.data[i] / input_scale) + input_zero_point;
      input->data.int8[i*loader.NUM_CHANNELS + 0] = std::max(-128, std::min(127, _quantized));

      _quantized = round(img.data[i + num_pixels] / input_scale) + input_zero_point;
      input->data.int8[i*loader.NUM_CHANNELS + 1] = std::max(-128, std::min(127, _quantized));

      _quantized = round(img.data[i + 2*num_pixels] / input_scale) + input_zero_point;
      input->data.int8[i*loader.NUM_CHANNELS + 2] = std::max(-128, std::min(127, _quantized));
    }
    TF_LITE_ENSURE_STATUS(interpreter.Invoke());

    std::vector<float> dequantized_output(NUM_CLASS);
    for (uint8_t i = 0; i < NUM_CLASS; i++) {
      dequantized_output[i] = (output->data.int8[i] - output_zero_point) * output_scale;
    }
    uint8_t y_pred = std::distance(dequantized_output.begin(), std::max_element(dequantized_output.begin(), dequantized_output.end()));
    if (y_pred == (uint8_t)img.label){
      accuracy += 1.0;
    }
  }
  accuracy /= NUM_SAMPLES;
  MicroPrintf("[int8 model] accuracy: %f", accuracy);

  return kTfLiteOk;
}

int main(int argc, char* argv[]) {
  tflite::InitializeTarget();

  TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency());
  
  // function call for inference of one sample
  MicroPrintf("\n~~~INFERENCE OF ONE SAMPLE~~~\n");
  TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInference());
  TF_LITE_ENSURE_STATUS(LoadQuantModelAndPerformInference());

  // function calll for inference of all samples
  MicroPrintf("\n~~~INFERENCE OF ALL SAMPLES~~~\n");
  TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInferenceForAllData());
  TF_LITE_ENSURE_STATUS(LoadQuantModelAndPerformInferenceForAllData());

  MicroPrintf("\n~~~ALL TESTS PASSED~~~\n");
  return kTfLiteOk;
}
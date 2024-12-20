# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""This script can create a quant(int8) model from the saved TF model.

Run:
Build the train.py script
`bazel build tensorflow/lite/micro/examples/hello_world:train`

The following command first creates the trained TF float model that we will quantize later
`bazel-bin/tensorflow/lite/micro/examples/hello_world/train --save_tf_model --save_dir=/tmp/float_model/`

Build the ptq.py script
`bazel build tensorflow/lite/micro/examples/mnist/quantization:ptq`

Then we can run the ptq script to convert the float model to quant model as follows.
Note that we are using the directory of the TF model as the source_model_dir here.
The quant model (named hello_world_int8.tflite) will be created inside the target_dir.
`bazel-bin/tensorflow/lite/micro/examples/hello_world/quantization/ptq  --source_model_dir=/tmp/float_model --target_dir=/tmp/quant_model/`
`
bazel-bin/tensorflow/lite/micro/examples/slsi.mnist_cnn/quantization/ptq\
   --source_model_dir=/workspace/241216-tflite-micro-quantization/tflite-micro/tensorflow/lite/micro/examples/slsi.mnist_cnn/models/mnist\
   --target_dir=/workspace/241216-tflite-micro-quantization/tflite-micro/tensorflow/lite/micro/examples/slsi.mnist_cnn/models/
`
"""
import math
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

FLAGS = flags.FLAGS

flags.DEFINE_string("source_model_dir", "/tmp/float_model/",
                    "the directory where the trained model can be found.")
flags.DEFINE_string("target_dir", "/tmp/quant_model",
                    "the directory to save the quant model.")


# Load and prepare the MNIST dataset
def load_mnist_data():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Reshape images to be suitable for the neural network
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    return (x_train, y_train), (x_test, y_test)


def save_tflite_model(tflite_model, target_dir, model_name):
  """save the converted tflite model
  Args:
      tflite_model (binary): the converted model in serialized format.
      save_dir (str): the save directory
      model_name (str): model name to be saved
  """
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)
  save_path = os.path.join(target_dir, model_name)
  with open(save_path, "wb") as f:
    f.write(tflite_model)
  logging.info("Tflite model saved to %s", target_dir)


def convert_quantized_tflite_model(source_model_dir, x_values):
  """Convert the save TF model to tflite model, then save it as .tflite
    flatbuffer format

    Args:
        source_model_dir (tf.keras.Model): the trained hello_world flaot Model dir
        x_train (numpy.array): list of the training data

    Returns:
        The converted model in serialized format.
  """

  # Convert the model to the TensorFlow Lite format with quantization
  def representative_dataset(num_samples=500):
    for i in range(num_samples):
      yield [tf.expand_dims(x_values[i], axis=0)]
      # yield [x_values[i].reshape(1, 1)]

  converter = tf.lite.TFLiteConverter.from_saved_model(source_model_dir)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8
  converter.inference_output_type = tf.int8
  converter.representative_dataset = representative_dataset
  tflite_model = converter.convert()
  return tflite_model


def main(_):
  (x_train, y_train), (x_test, y_test) = load_mnist_data()

  # get data for calibration
  # shuffle the x_train and get the first 100 values
  x_values = x_train
  np.random.shuffle(x_values)
  x_values = x_values[:500].astype(np.float32)
  
  quantized_tflite_model = convert_quantized_tflite_model(
      FLAGS.source_model_dir, x_values)
  
  save_tflite_model(quantized_tflite_model,
                    FLAGS.target_dir,
                    model_name="mnist_int8.tflite")

if __name__ == "__main__":
  app.run(main)

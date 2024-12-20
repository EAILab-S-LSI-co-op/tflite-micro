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
"""hellow_world model training for sinwave recognition

Run:
`bazel build tensorflow/lite/micro/examples/mnist:train`
`bazel-bin/tensorflow/lite/micro/examples/mnist/train --save_tf_model --save_dir=/workspace/241216-tflite-micro-quantization/tflite-micro/tensorflow/lite/micro/examples/slsi.mnist_cnn/models`
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

flags.DEFINE_integer("epochs", 500, "number of epochs to train the model.")
flags.DEFINE_string("save_dir", "/tmp/hello_world_models",
                    "the directory to save the trained model.")
flags.DEFINE_boolean("save_tf_model", False,
                     "store the original unconverted tf model.")


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

# Build the neural network model
def create_model():
    model = keras.Sequential([
        # Convolutional layers for feature extraction
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Flatten the 2D feature maps for fully connected layers
        keras.layers.Flatten(),
        
        # Fully connected layers for classification
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Train the model
def train_model(model, x_train, y_train, x_test, y_test, epochs=10):
    # Train the model
    history = model.fit(x_train, y_train, 
                        epochs=epochs, 
                        validation_split=0.2,
                        batch_size=512)
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_accuracy * 100:.2f}%')
    
    return history

def convert_tflite_model(model):
  """Convert the save TF model to tflite model, then save it as .tflite flatbuffer format
    Args:
        model (tf.keras.Model): the trained hello_world Model
    Returns:
        The converted model in serialized format.
  """
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  return tflite_model


def save_tflite_model(tflite_model, save_dir, model_name):
  """save the converted tflite model
  Args:
      tflite_model (binary): the converted model in serialized format.
      save_dir (str): the save directory
      model_name (str): model name to be saved
  """
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  save_path = os.path.join(save_dir, model_name)
  with open(save_path, "wb") as f:
    f.write(tflite_model)
  logging.info("Tflite model saved to %s", save_dir)


def main(_):
  (x_train, y_train), (x_test, y_test) = load_mnist_data()
  
  model = create_model()
  history = train_model(model, x_train, y_train, x_test, y_test, epochs=20)
  
  # save the original tf model
  model.export(f"{FLAGS.save_dir}/mnist")
  model.save(f"{FLAGS.save_dir}/mnist.h5")
  logging.info("TF model saved to %s", FLAGS.save_dir)
  
  # Convert and save the model to .tflite
  tflite_model = convert_tflite_model(model)
  save_tflite_model(tflite_model,
                    FLAGS.save_dir,
                    model_name="mnist.tflite")


if __name__ == "__main__":
  app.run(main)
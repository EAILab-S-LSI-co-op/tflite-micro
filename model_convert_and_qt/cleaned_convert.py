import numpy as np

import torch
import torch.nn as nn 
import torch.onnx
import torchvision.models as models
import onnx
import onnxruntime
import torch_pruning as tp
from model import Model
import tensorflow as tf

from onnx_tf.backend import prepare
import argparse
import random 
import copy
def fix_seeds(seed=42):
    '''Fix the seeds for reproducibility'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prune_model(model,args):
    
    '''Prune the model'''
    
    dep_graph = tp.DependencyGraph().build_dependency(model, torch.randn(1, 1, 28, 28) )
    
    if args.prune_method == 'global':
        example_inputs = torch.randn(1, 1, 28,28)

        # 1. Importance criterion, here we calculate the L2 Norm of grouped weights as the importance score
        imp = tp.importance.GroupNormImportance(p=2) 

        # 2. Initialize a pruner with the model and the importance criterion
        ignored_layers = []
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == 10:
                ignored_layers.append(m) # DO NOT prune the final classifier!

        pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
            model,
            example_inputs,
            importance=imp,
            pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers,
            round_to=8, # It's recommended to round dims/channels to 4x or 8x for acceleration. Please see: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
        )

        # 3. Prune the model
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")
    elif args.prune_method == 'test_prune':
        group = dep_graph.get_pruning_group(model.conv1,tp.prune_conv_out_channels,idxs=[0,1,2,3,4,5])

        if dep_graph.check_pruning_group(group):
            group.prune()
    
    prune_results = base_macs, base_nparams, macs, nparams
    return prune_results
    


def convert_model_to_onnx_mnist(model, onnx_filename):
    '''Convert the model to ONNX format'''
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(model, dummy_input, 
                      onnx_filename, 
                      verbose=True,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names = ['input'],
                        output_names = ['output'],
                      )
    
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)



def load_mnist_data():
    ''' bring random data for quantization'''
        # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Reshape images to be suitable for the neural network
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    return (x_train, y_train), (x_test, y_test)


def representative_datasets_gen(sample_count=1000):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Reshape images to be suitable for the neural network
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_values = x_train.astype(np.float32)
    for i in range(sample_count):
        yield [tf.expand_dims(x_values[i], 0)]
    


def convert_onnx_to_tf(onnx_filename, tf_filename):
    '''Convert the model to TensorFlow format'''
    onnx_model = onnx.load(onnx_filename)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_filename)
    
def convert_tf_to_tflite(tf_filename, tflite_filename):
    '''Convert the model to TensorFlow Lite format'''
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_filename)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8
    # converter.representative_dataset = representative_datasets_gen
    tflite_model = converter.convert()

    
    
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)


def load_mnist_data():
    '''##Load the MNIST dataset'''
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train[..., tf.newaxis].astype("float32"), x_test[..., tf.newaxis].astype("float32")
    
    return (x_train, y_train), (x_test, y_test)


def invoke_tflm_interpreter(input_shape, interpreter, x_value, input_index, output_index):
    '''##Invoke the TFLM interpreter'''
    input_data = np.reshape(x_value, input_shape)
    interpreter.set_input(input_data, input_index)
    interpreter.invoke()
    y_quantized = np.reshape(interpreter.get_output(output_index), -1)
    return y_quantized

def invoke_tflite_interpreter(input_shape, interpreter, x_value, input_index, output_index):
    '''##Invoke the TFLite interpreter'''
    input_data = np.reshape(x_value, input_shape)
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_index)
    y_quantized = np.reshape(tflite_output, -1)
    return y_quantized

def get_tflm_prediction(model_path, x_values):
    '''##Get the TFLM prediction'''
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    input_shape = input_details[0]['shape']
    y_values = []
    for x_value in x_values:
        y_values.append(invoke_tflite_interpreter(input_shape, interpreter, x_value, input_index, output_index))
    return y_values

def compare_onnx_torch(onnx_filename, model, x_values):
    '''##Compare the ONNX and PyTorch models'''
    ort_session = onnxruntime.InferenceSession(onnx_filename)
    ort_inputs = {ort_session.get_inputs()[0].name: x_values}
    ort_outs = ort_session.run(None, ort_inputs)
    with torch.no_grad():
        model_out = model(torch.tensor(x_values))
    print(ort_outs, model_out)
    np.testing.assert_allclose(ort_outs[0], model_out.numpy(), rtol=1e-02, atol=1e-03)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def evaluate_model(model, x_test, y_test):
    '''##Evaluate the model'''
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(x_test)):
            x_test_now = x_test[i].reshape(1, 1, 28, 28)
            output = model(torch.tensor(x_test_now))
            correct += (torch.argmax(output) == y_test[i]).item()
            total += 1
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    return accuracy

def main(args):
    model = Model()
    model.load_state_dict(torch.load('mnist_cnn.pth'))
    
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    ori_model = copy.deepcopy(model)
    # prune_model(model)
    onnx_filename = 'mnist.onnx'
    tf_filename = 'mnist_tf'
    tflite_filename = 'mnist.tflite'
    
    if args.prune:
        prune_results = prune_model(model,args)
        pruned_model = copy.deepcopy(model)
        
    convert_model_to_onnx_mnist(model, onnx_filename)
    compare_onnx_torch(onnx_filename, model, np.random.rand(1, 1, 28, 28).astype(np.float32))
    convert_onnx_to_tf(onnx_filename, tf_filename)
    convert_tf_to_tflite(tf_filename, tflite_filename)
    
    x_test, y_test = x_test.astype(np.float32), y_test.astype(np.float32)
    
    last_output = get_tflm_prediction(tflite_filename, x_test)
    acc = np.mean(np.argmax(last_output, axis=1) == y_test)
    
    base_model = Model()
    base_model.load_state_dict(torch.load('mnist_cnn.pth'))
    base_acc = evaluate_model(base_model, x_test, y_test)
    
    pruned_model.eval()
    pruned_acc = evaluate_model(pruned_model, x_test, y_test)
        
    print(f'Before Prune Base Model Accuracy: {base_acc / len(x_test) * 100:.4f}%')
    print(f'After Prune Base Model Accuracy: {pruned_acc / len(x_test) * 100:.4f}%')
    print(f'After Prune TFLite Model Accuracy: {acc * 100:.4f}%')
    
    if args.prune:
        print(f'Prune Results')
        print(f'Base MACs: {prune_results[0]/1e9} G, Base #Params: {prune_results[1]/1e6} M')
        print(f'Pruned MACs: {prune_results[2]/1e9} G, Pruned #Params: {prune_results[3]/1e6} M')
        # percentage
        print(f'MACs Pruned: {100 * (prune_results[0] - prune_results[2]) / prune_results[0]:.2f}%')
        print(f'#Params Pruned: {100 * (prune_results[1] - prune_results[3]) / prune_results[1]:.2f}%')
        
    
    
    
if __name__ == '__main__':
    fix_seeds(21)
    args = argparse.ArgumentParser()
    args.add_argument('--save_dir', type=str, default='saved_models')
    args.add_argument('--prune', type=bool, default=True)
    args.add_argument('--prune_method', type=str, default='global')
    args = args.parse_args()
    main(args)
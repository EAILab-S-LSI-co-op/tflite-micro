load("//tensorflow/lite/micro:build_def.bzl", "generate_cc_arrays")

package(
    default_visibility = ["//visibility:public"],
)

exports_files(
    srcs = [
        "mnist_float.tflite",
        "mnist_int8.tflite",
    ],
    visibility = [
        "//codegen/examples/mnist:__subpackages__",
        "//tensorflow/lite/micro/examples/mnist:__subpackages__",
    ],
)

generate_cc_arrays(
    name = "generated_mnist_float_model_cc",
    src = "mnist_float.tflite",
    out = "mnist_float_model_data.cc",
)

generate_cc_arrays(
    name = "generated_mnist_float_model_hdr",
    src = "mnist_float.tflite",
    out = "mnist_float_model_data.h",
)

generate_cc_arrays(
    name = "generated_mnist_int8_model_cc",
    src = "mnist_int8.tflite",
    out = "mnist_int8_model_data.cc",
)

generate_cc_arrays(
    name = "generated_mnist_int8_model_hdr",
    src = "mnist_int8.tflite",
    out = "mnist_int8_model_data.h",
)

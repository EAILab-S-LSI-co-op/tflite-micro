load("//tensorflow/lite/micro:build_def.bzl", "generate_cc_arrays")

package(
    default_visibility = ["//visibility:public"],
)

exports_files(
    srcs = [
        "cifar10_float.tflite",
        "cifar10_int8.tflite",
    ],
    visibility = [
        "//codegen/examples/cifar10:__subpackages__",
        "//tensorflow/lite/micro/examples/cifar10:__subpackages__",
    ],
)

generate_cc_arrays(
    name = "generated_cifar10_float_model_cc",
    src = "cifar10_float.tflite",
    out = "cifar10_float_model_data.cc",
)

generate_cc_arrays(
    name = "generated_cifar10_float_model_hdr",
    src = "cifar10_float.tflite",
    out = "cifar10_float_model_data.h",
)

generate_cc_arrays(
    name = "generated_cifar10_int8_model_cc",
    src = "cifar10_int8.tflite",
    out = "cifar10_int8_model_data.cc",
)

generate_cc_arrays(
    name = "generated_cifar10_int8_model_hdr",
    src = "cifar10_int8.tflite",
    out = "cifar10_int8_model_data.h",
)

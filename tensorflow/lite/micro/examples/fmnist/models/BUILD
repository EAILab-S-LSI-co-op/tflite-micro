load("//tensorflow/lite/micro:build_def.bzl", "generate_cc_arrays")

package(
    default_visibility = ["//visibility:public"],
)

exports_files(
    srcs = [
        "fmnist_float.tflite",
        "fmnist_int8.tflite",
    ],
    visibility = [
        "//codegen/examples/fmnist:__subpackages__",
        "//tensorflow/lite/micro/examples/fmnist:__subpackages__",
    ],
)

generate_cc_arrays(
    name = "generated_fmnist_float_model_cc",
    src = "fmnist_float.tflite",
    out = "fmnist_float_model_data.cc",
)

generate_cc_arrays(
    name = "generated_fmnist_float_model_hdr",
    src = "fmnist_float.tflite",
    out = "fmnist_float_model_data.h",
)

generate_cc_arrays(
    name = "generated_fmnist_int8_model_cc",
    src = "fmnist_int8.tflite",
    out = "fmnist_int8_model_data.cc",
)

generate_cc_arrays(
    name = "generated_fmnist_int8_model_hdr",
    src = "fmnist_int8.tflite",
    out = "fmnist_int8_model_data.h",
)

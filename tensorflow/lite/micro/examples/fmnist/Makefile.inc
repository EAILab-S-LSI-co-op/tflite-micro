MNIST_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/fmnist/fmnist_test.cc

MNIST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/fmnist/fmnist_test.cc

MNIST_HDRS :=

MNIST_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/fmnist/models/fmnist_float.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/fmnist/models/fmnist_int8.tflite

MNIST_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/fmnist/models/fmnist_float_model_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/fmnist/models/fmnist_int8_model_data.cc

MNIST_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/fmnist/models/fmnist_float_model_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/fmnist/models/fmnist_int8_model_data.h

# Tests loading and running the sine model.
$(eval $(call microlite_test,fmnist_test,\
$(MNIST_TEST_SRCS),,$(MNIST_GENERATOR_INPUTS)))

# Builds a standalone binary.
$(eval $(call microlite_test,fmnist,\
$(MNIST_SRCS),,$(MNIST_GENERATOR_INPUTS)))

# Add sources and headers generated from $(MNIST_GENERATOR_INPUTS).
MNIST_SRCS += $(MNIST_GENERATED_SRCS)
MNIST_HDRS += $(MNIST_GENERATED_HDRS)

list_fmnist_example_sources:
	@echo $(MNIST_SRCS)

list_fmnist_example_headers:
	@echo $(MNIST_HDRS)

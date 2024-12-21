CIFAR10_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/cifar10/cifar10_test.cc

CIFAR10_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/cifar10/cifar10_test.cc

CIFAR10_HDRS :=

CIFAR10_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/cifar10/models/cifar10_float.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/cifar10/models/cifar10_int8.tflite

CIFAR10_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/cifar10/models/cifar10_float_model_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/cifar10/models/cifar10_int8_model_data.cc

CIFAR10_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/cifar10/models/cifar10_float_model_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/cifar10/models/cifar10_int8_model_data.h

# Tests loading and running the sine model.
$(eval $(call microlite_test,cifar10_test,\
$(CIFAR10_TEST_SRCS),,$(CIFAR10_GENERATOR_INPUTS)))

# Builds a standalone binary.
$(eval $(call microlite_test,cifar10,\
$(CIFAR10_SRCS),,$(CIFAR10_GENERATOR_INPUTS)))

# Add sources and headers generated from $(CIFAR10_GENERATOR_INPUTS).
CIFAR10_SRCS += $(CIFAR10_GENERATED_SRCS)
CIFAR10_HDRS += $(CIFAR10_GENERATED_HDRS)

list_cifar10_example_sources:
	@echo $(CIFAR10_SRCS)

list_cifar10_example_headers:
	@echo $(CIFAR10_HDRS)

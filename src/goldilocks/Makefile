TARGET_EXEC := example

BUILD_DIR := ./build
SRC_DIRS := ./src
TEST_DIRS := ./test

LIBOMP := $(shell find /usr/lib/llvm-* -name "libomp.so" | sed 's/libomp.so//')
ifndef LIBOMP
$(error LIBOMP is not set, you need to install libomp-dev)
endif

#CXX := mpiCC
CXX = g++
CXXFLAGS := -std=c++17 -Wall -pthread -fopenmp
LDFLAGS := -lpthread -lgmp -lstdc++ -lomp -lgmpxx -lbenchmark -L$(LIBOMP)
ASFLAGS := -felf64 

# Debug build flags
ifeq ($(dbg),1)
      CXXFLAGS += -g
else
      CXXFLAGS += -O3
endif

### Establish the operating system name
KERNEL = $(shell uname -s)
ifneq ($(KERNEL),Linux)
 $(error "$(KERNEL), is not a valid kernel")
endif
ARCH = $(shell uname -m)
ifneq ($(ARCH),x86_64)
 $(error "$(ARCH), is not a valid architecture")
endif

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.asm)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) $(CXXFLAGS) -o $@ $(LDFLAGS)

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cc.o: %.cc
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p
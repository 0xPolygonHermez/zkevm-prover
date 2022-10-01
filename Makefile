TARGET_ZKP := zkProver
TARGET_BCT := bctree

BUILD_DIR := ./build
SRC_DIRS := ./src ./test ./tools

LIBOMP := $(shell find /usr/lib/llvm-* -name "libomp.so" | sed 's/libomp.so//')
ifndef LIBOMP
$(error LIBOMP is not set, you need to install libomp-dev)
endif

CXX := g++
AS := nasm
CXXFLAGS := -std=c++17 -Wall -pthread -flarge-source-files -Wno-unused-label -rdynamic -mavx2
LDFLAGS := -lprotobuf -lsodium -lgrpc -lgrpc++ -lgrpc++_reflection -lgpr -lpthread -lpqxx -lpq -lgmp -lstdc++ -lomp -lgmpxx -lsecp256k1 -lcrypto -luuid -L$(LIBOMP)
CFLAGS := -fopenmp -D'memset_s(W,WL,V,OL)=memset(W,V,OL)'
ASFLAGS := -felf64

# Debug build flags
ifeq ($(dbg),1)
      CXXFLAGS += -g
else
      CXXFLAGS += -O3
endif

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP

SRCS_ZKP := $(shell find $(SRC_DIRS) ! -path "./tools/starkpil/bctree/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/tests/*" -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_ZKP := $(SRCS_ZKP:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

SRCS_BCT := $(shell find $(SRC_DIRS) ! -path "./src/main.cpp" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/tests/*" -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_BCT := $(SRCS_BCT:%=$(BUILD_DIR)/%.o)

all: $(BUILD_DIR)/$(TARGET_ZKP)

bctree: $(BUILD_DIR)/$(TARGET_BCT)

$(BUILD_DIR)/$(TARGET_ZKP): $(OBJS_ZKP)
	$(CXX) $(OBJS_ZKP) $(CXXFLAGS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/$(TARGET_BCT): $(OBJS_BCT)
	$(CXX) $(OBJS_BCT) $(CXXFLAGS) -o $@ $(LDFLAGS)

# assembly
$(BUILD_DIR)/%.asm.o: %.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cc.o: %.cc
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p

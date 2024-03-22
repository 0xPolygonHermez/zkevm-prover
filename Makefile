TARGET_ZKP := zkProver
TARGET_BCT := bctree
TARGET_MNG += mainGenerator
TARGET_PLG += polsGenerator
TARGET_PLD += polsDiff
TARGET_TEST := zkProverTest
TARGET_SETUP := fflonkSetup

BUILD_DIR := ./build
SRC_DIRS := ./src ./test ./tools
SETUP_DIRS := ./src/rapidsnark
SETUP_DPNDS_DIR := src/ffiasm

GRPCPP_FLAGS := $(shell pkg-config grpc++ --cflags)
GRPCPP_LIBS := $(shell pkg-config grpc++ --libs) -lgrpc++_reflection
ifndef GRPCPP_LIBS
$(error gRPC++ could not be found via pkg-config, you need to install them)
endif

CXX := g++
AS := nasm
CXXFLAGS := -std=c++17 -Wall -pthread -flarge-source-files -Wno-unused-label -rdynamic -mavx2 $(GRPCPP_FLAGS) #-Wfatal-errors
LDFLAGS := -lprotobuf -lsodium -lgpr -lpthread -lpqxx -lpq -lgmp -lstdc++ -lgmpxx -lsecp256k1 -lcrypto -luuid -fopenmp -liomp5 $(GRPCPP_LIBS)
CFLAGS := -fopenmp
ASFLAGS := -felf64

# Debug build flags
ifeq ($(dbg),1)
      CXXFLAGS += -g -D DEBUG
else
      CXXFLAGS += -O3
endif

# Verify if AVX-512 is supported
# for now disabled, to enable it, you only need to uncomment these lines
#AVX512_SUPPORTED := $(shell cat /proc/cpuinfo | grep -E 'avx512' -m 1)

#ifneq ($(AVX512_SUPPORTED),)
#	CXXFLAGS += -mavx512f -D__AVX512__
#endif

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP

GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

INC_DIRS := $(shell find $(SRC_DIRS) -type d) $(sort $(dir))
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

SRCS_ZKP := $(shell find $(SRC_DIRS) ! -path "./src/fflonk_setup/fflonk_setup*" ! -path "./tools/starkpil/bctree/*" ! -path "./test/examples/*" ! -path "./test/prover/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/tests/*" ! -path "./src/main_generator/*" ! -path "./src/pols_generator/*" ! -path "./src/pols_diff/*" -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_ZKP := $(SRCS_ZKP:%=$(BUILD_DIR)/%.o)
DEPS_ZKP := $(OBJS_ZKP:.o=.d)

SRCS_BCT := $(shell find ./tools/starkpil/bctree/build_const_tree.cpp ./tools/starkpil/bctree/main.cpp ./src/goldilocks/src ./src/starkpil/merkleTree/merkleTreeBN128.cpp ./src/starkpil/merkleTree/merkleTreeGL.cpp ./src/poseidon_opt/poseidon_opt.cpp ./src/XKCP ./src/ffiasm ./src/starkpil/stark_info.* ./src/utils/* -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_BCT := $(SRCS_BCT:%=$(BUILD_DIR)/%.o)
DEPS_BCT := $(OBJS_BCT:.o=.d)

SRCS_TEST := $(shell find ./test/examples/ ./src/XKCP ./src/goldilocks/src ./src/starkpil/stark_info.* ./src/starkpil/starks.* ./src/starkpil/chelpers.* ./src/rapidsnark/binfile_utils.* ./src/starkpil/steps.* ./src/starkpil/polinomial.hpp ./src/starkpil/merkleTree/merkleTreeGL.* ./src/starkpil/transcript/transcript.* ./src/starkpil/fri ./src/ffiasm ./src/utils ! -path "./src/starkpil/fri/friProveC12.*" -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_TEST := $(SRCS_TEST:%=$(BUILD_DIR)/%.o)
DEPS_TEST := $(OBJS_TEST:.o=.d)

SRCS_SETUP := $(shell find $(SETUP_DIRS) ! -path "./src/sm/*" ! -path "./src/main_sm/*" -name *.cpp)
SRCS_SETUP += $(shell find src/XKCP -name *.cpp)
SRCS_SETUP += $(shell find src/fflonk_setup -name fflonk_setup.cpp)
SRCS_SETUP += $(addprefix $(SETUP_DPNDS_DIR)/, alt_bn128.cpp fr.cpp fq.cpp fnec.cpp fec.cpp misc.cpp naf.cpp splitparstr.cpp)
SRCS_SETUP += $(shell find $(SETUP_DPNDS_DIR) -name *.asm)
OBJS_SETUP := $(patsubst %,$(BUILD_DIR)/%.o,$(SRCS_SETUP))
OBJS_SETUP := $(filter-out $(BUILD_DIR)/src/main.cpp.o, $(OBJS_SETUP)) # Exclude main.cpp from test build
OBJS_SETUP := $(filter-out $(BUILD_DIR)/src/main_test.cpp.o, $(OBJS_SETUP)) # Exclude main.cpp from test build
DEPS_SETUP := $(OBJS_SETUP:.o=.d)

all: $(BUILD_DIR)/$(TARGET_ZKP)

bctree: $(BUILD_DIR)/$(TARGET_BCT)

test: $(BUILD_DIR)/$(TARGET_TEST)

$(BUILD_DIR)/$(TARGET_ZKP): $(OBJS_ZKP)
	$(CXX) $(OBJS_ZKP) $(CXXFLAGS) -o $@ $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)

$(BUILD_DIR)/$(TARGET_BCT): $(OBJS_BCT)
	$(CXX) $(OBJS_BCT) $(CXXFLAGS) -o $@ $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)

$(BUILD_DIR)/$(TARGET_TEST): $(OBJS_TEST)
	$(CXX) $(OBJS_TEST) $(CXXFLAGS) -o $@ $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)

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

main_generator: $(BUILD_DIR)/$(TARGET_MNG)

$(BUILD_DIR)/$(TARGET_MNG): ./src/main_generator/main_generator.cpp
	$(MKDIR_P) $(BUILD_DIR)
	g++ -g ./src/main_generator/main_generator.cpp -o $@ -lgmp

generate: main_generator
	$(BUILD_DIR)/$(TARGET_MNG) all

pols_generator: $(BUILD_DIR)/$(TARGET_PLG)

$(BUILD_DIR)/$(TARGET_PLG): ./src/pols_generator/pols_generator.cpp
	$(MKDIR_P) $(BUILD_DIR)
	g++ -g ./src/pols_generator/pols_generator.cpp -o $@ -lgmp

pols_diff: $(BUILD_DIR)/$(TARGET_PLD)

$(BUILD_DIR)/$(TARGET_PLD): ./src/pols_diff/pols_diff.cpp
	$(MKDIR_P) $(BUILD_DIR)
	g++ -g ./src/pols_diff/pols_diff.cpp $(CXXFLAGS) $(INC_FLAGS) -o $@ $(LDFLAGS) 

fflonk_setup: $(BUILD_DIR)/$(TARGET_SETUP)

$(BUILD_DIR)/$(TARGET_SETUP): $(OBJS_SETUP)
	$(CXX) $(OBJS_SETUP) $(CXXFLAGS) $(LDFLAGS) -o $@

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)
	find . -name main_exec_generated*pp -delete

-include $(DEPS_ZKP)
-include $(DEPS_SETUP)
-include $(DEPS_BCT)

MKDIR_P ?= mkdir -p

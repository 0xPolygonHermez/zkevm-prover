#INFO := $(shell cd src/goldilocks && ./configure.sh && cd ../.. && sleep 2)
include src/goldilocks/CudaArch.mk
NVCC := /usr/local/cuda/bin/nvcc

TARGET_ZKP := zkProver
TARGET_ZKP_GPU := zkProverGpu
TARGET_BCT := bctree
TARGET_MNG := mainGenerator
TARGET_MNG_10 := mainGenerator10
TARGET_PLG := polsGenerator
TARGET_PLD := polsDiff
TARGET_TEST := zkProverTest
TARGET_TEST_GPU := zkProverTestGpu
TARGET_W2DB := witness2db
TARGET_EXPRESSIONS := zkProverExpressions
TARGET_SETUP := fflonkSetup

BUILD_DIR := ./build
BUILD_DIR_GPU := ./build-gpu
SRC_DIRS := ./src ./test ./tools
SETUP_DIRS := ./src/rapidsnark

GRPCPP_FLAGS := $(shell pkg-config grpc++ --cflags)
GRPCPP_LIBS := $(shell pkg-config grpc++ --libs) -lgrpc++_reflection
ifndef GRPCPP_LIBS
$(error gRPC++ could not be found via pkg-config, you need to install them)
endif

CXX := g++
AS := nasm
CXXFLAGS := -std=c++17 -Wall -pthread -flarge-source-files -Wno-unused-label -rdynamic -mavx2 $(GRPCPP_FLAGS) #-Wfatal-errors

LDFLAGS_GPU := -lprotobuf -lsodium -lgpr -lpthread -lpqxx -lpq -lgmp -lstdc++ -lgmpxx -lsecp256k1 -lcrypto -luuid -liomp5 $(GRPCPP_LIBS)
LDFLAGS := $(LDFLAGS_GPU) -fopenmp
CXXFLAGS_W2DB := -std=c++17 -Wall -pthread -flarge-source-files -Wno-unused-label -rdynamic -mavx2
LDFLAGS_W2DB := -lgmp -lstdc++ -lgmpxx

CFLAGS := -fopenmp
ASFLAGS := -felf64

# Debug build flags
ifeq ($(dbg),1)
      CXXFLAGS += -g -D DEBUG
else
      CXXFLAGS += -O3
endif

ifdef PROVER_FORK_ID
	  CXXFLAGS += -DPROVER_FORK_ID=$(PROVER_FORK_ID)
endif

ifneq ($(avx512),0)
ifeq ($(avx512),1)
	CXXFLAGS += -mavx512f -D__AVX512__
else
# check if AVX-512 is supported
AVX512_SUPPORTED := $(shell cat /proc/cpuinfo | grep -E 'avx512' -m 1)
ifneq ($(AVX512_SUPPORTED),)
	CXXFLAGS += -mavx512f -D__AVX512__
endif
endif
endif

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP

GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

INC_DIRS := $(shell find $(SRC_DIRS) -type d) $(sort $(dir))
INC_FLAGS := $(addprefix -I,$(INC_DIRS)) -I/usr/local/cuda/include

SRCS_ZKP := $(shell find $(SRC_DIRS) ! -path "./src/fflonk_setup/fflonk_setup*" ! -path "./tools/starkpil/bctree/*" ! -path "./test/examples/*" ! -path "./test/expressions/*" ! -path "./test/prover/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/tests/*" ! -path "./src/main_generator/*" ! -path "./src/pols_generator/*" ! -path "./src/pols_diff/*" ! -path "./src/witness2db/*" \( -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc \))
SRCS_ZKP_GPU := $(shell find $(SRC_DIRS) ! -path "./src/fflonk_setup/fflonk_setup*" ! -path "./tools/starkpil/bctree/*" ! -path "./test/examples/*" ! -path "./test/expressions/*"  ! -path "./test/prover/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/tests/*" ! -path "./src/main_generator/*" ! -path "./src/pols_generator/*" ! -path "./src/pols_diff/*" ! -path "./src/witness2db/*" ! -path "./src/goldilocks/utils/deviceQuery.cu" \( -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc -or -name *.cu \))

OBJS_ZKP := $(SRCS_ZKP:%=$(BUILD_DIR)/%.o)
OBJS_ZKP_GPU := $(SRCS_ZKP_GPU:%=$(BUILD_DIR_GPU)/%.o)
DEPS_ZKP := $(OBJS_ZKP:.o=.d)

SRCS_BCT := $(shell find ./tools/starkpil/bctree/build_const_tree.cpp ./tools/starkpil/bctree/main.cpp ./src/goldilocks/src ./src/starkpil/merkleTree/merkleTreeBN128.cpp ./src/starkpil/merkleTree/merkleTreeGL.cpp ./src/poseidon_opt/poseidon_opt.cpp ./src/XKCP ./src/ffiasm ./src/starkpil/stark_info.* ./src/utils/* \( -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc \))
OBJS_BCT := $(SRCS_BCT:%=$(BUILD_DIR)/%.o)
DEPS_BCT := $(OBJS_BCT:.o=.d)

SRCS_TEST := $(shell find ./test/examples/ ./src/XKCP ./src/goldilocks/src ./src/starkpil/stark_info.* ./src/starkpil/starks.* ./src/starkpil/chelpers.* ./src/rapidsnark/binfile_utils.* ./src/starkpil/steps.* ./src/starkpil/polinomial.hpp ./src/starkpil/merkleTree/merkleTreeGL.* ./src/starkpil/transcript/transcript.* ./src/starkpil/fri ./src/ffiasm ./src/utils ./tools/sm/sha256/sha256.cpp ./tools/sm/sha256/bcon/bcon_sha256.cpp ! -path "./src/starkpil/fri/friProveC12.*" \( -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc \))

SRCS_TEST_GPU := $(shell find ./test/examples/ ./src/XKCP ./src/goldilocks/src ./src/goldilocks/utils ./src/starkpil/stark_info.* ./src/starkpil/starks.* ./src/starkpil/chelpers.*  ./src/starkpil/chelpers_steps_gpu.cu  ./src/rapidsnark/binfile_utils.* ./src/starkpil/steps.* ./src/starkpil/polinomial.hpp ./src/starkpil/merkleTree/merkleTreeGL.* ./src/starkpil/transcript/transcript.* ./src/starkpil/fri ./src/ffiasm ./src/utils ./tools/sm/sha256/sha256.cpp ./tools/sm/sha256/bcon/bcon_sha256.cpp ! -path "./src/starkpil/fri/friProveC12.*" ! -path "./src/goldilocks/utils/deviceQuery.cu" \( -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc -or -name *.cu \))
OBJS_TEST := $(SRCS_TEST:%=$(BUILD_DIR)/%.o)
OBJS_TEST_GPU := $(SRCS_TEST_GPU:%=$(BUILD_DIR_GPU)/%.o)
DEPS_TEST := $(OBJS_TEST:.o=.d)

SRCS_W2DB := ./src/witness2db/witness2db.cpp  ./src/goldilocks/src/goldilocks_base_field.cpp ./src/goldilocks/src/poseidon_goldilocks.cpp
OBJS_W2DB := $(SRCS_W2DB:%=$(BUILD_DIR)/%.o)
DEPS_W2DB := $(OBJS_W2DB:.o=.d)

SRCS_EXPRESSIONS := $(shell find ./test/expressions/ ./src/XKCP ./src/goldilocks/src ./src/starkpil/stark_info.*  ./src/starkpil/chelpers.* ./src/rapidsnark/binfile_utils.*  ./src/starkpil/steps.* ./src/starkpil/polinomial.hpp ./src/ffiasm ./src/utils ! -path "./src/starkpil/fri/friProveC12.*"  \( -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc \))
OBJS_EXPRESSIONS := $(SRCS_EXPRESSIONS:%=$(BUILD_DIR)/%.o)
DEPS_EXPRESSIONS := $(OBJS_EXPRESSIONS:.o=.d)

SRCS_SETUP := $(shell find $(SETUP_DIRS) ! -path "./src/sm/*" ! -path "./src/main_sm/*" -name *.cpp)
SRCS_SETUP += $(shell find src/XKCP -name *.cpp)
SRCS_SETUP += $(shell find src/fflonk_setup -name fflonk_setup.cpp)
SRCS_SETUP += $(shell find src/ffiasm/* -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_SETUP := $(patsubst %,$(BUILD_DIR)/%.o,$(SRCS_SETUP))
OBJS_SETUP := $(filter-out $(BUILD_DIR)/src/main.cpp.o, $(OBJS_SETUP)) # Exclude main.cpp from test build
OBJS_SETUP := $(filter-out $(BUILD_DIR)/src/main_test.cpp.o, $(OBJS_SETUP)) # Exclude main.cpp from test build
DEPS_SETUP := $(OBJS_SETUP:.o=.d)

cpu: $(BUILD_DIR)/$(TARGET_ZKP)
gpu: $(BUILD_DIR_GPU)/$(TARGET_ZKP_GPU)

bctree: $(BUILD_DIR)/$(TARGET_BCT)

test: $(BUILD_DIR)/$(TARGET_TEST)
test_gpu:  $(BUILD_DIR_GPU)/$(TARGET_TEST_GPU)

expressions: ${BUILD_DIR}/$(TARGET_EXPRESSIONS)

$(BUILD_DIR)/$(TARGET_ZKP): $(OBJS_ZKP)
	$(CXX) $(OBJS_ZKP) $(CXXFLAGS) -o $@ $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)

$(BUILD_DIR_GPU)/$(TARGET_ZKP_GPU): $(OBJS_ZKP_GPU)
	$(NVCC) $(OBJS_ZKP_GPU) -O3 -arch=$(CUDA_ARCH) -o $@ $(LDFLAGS_GPU)

$(BUILD_DIR)/$(TARGET_BCT): $(OBJS_BCT)
	$(CXX) $(OBJS_BCT) $(CXXFLAGS) -o $@ $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)

$(BUILD_DIR)/$(TARGET_TEST): $(OBJS_TEST)
	$(CXX) $(OBJS_TEST) $(CXXFLAGS) -o $@ $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)

$(BUILD_DIR_GPU)/$(TARGET_TEST_GPU): $(OBJS_TEST_GPU)
	$(NVCC) $(OBJS_TEST_GPU) -O3 -arch=$(CUDA_ARCH) -o $@ $(LDFLAGS_GPU)

$(BUILD_DIR)/$(TARGET_EXPRESSIONS): $(OBJS_EXPRESSIONS)
	$(CXX) $(OBJS_EXPRESSIONS) $(CXXFLAGS) -o $@ $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)

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

# assembly
$(BUILD_DIR_GPU)/%.asm.o: %.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

# c++ source
$(BUILD_DIR_GPU)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ -DENABLE_EXPERIMENTAL_CODE $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR_GPU)/%.cc.o: %.cc
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ -DENABLE_EXPERIMENTAL_CODE $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# cuda source
$(BUILD_DIR_GPU)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) -D__USE_CUDA__ -DENABLE_EXPERIMENTAL_CODE $(INC_FLAGS) -Isrc/goldilocks/utils -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 -Xcompiler -O3 -O3 -arch=$(CUDA_ARCH) -O3 $< -dc --output-file $@

main_generator: $(BUILD_DIR)/$(TARGET_MNG)

$(BUILD_DIR)/$(TARGET_MNG): ./src/main_generator/main_generator.cpp ./src/config/definitions.hpp
	$(MKDIR_P) $(BUILD_DIR)
	g++ -g ./src/main_generator/main_generator.cpp -o $@ -lgmp

main_generator_10: $(BUILD_DIR)/$(TARGET_MNG_10)

$(BUILD_DIR)/$(TARGET_MNG_10): ./src/main_generator/main_generator_10.cpp ./src/config/definitions.hpp
	$(MKDIR_P) $(BUILD_DIR)
	g++ -g $(CXXFLAGS) ./src/main_generator/main_generator_10.cpp ./src/config/fork_info.cpp -o $@ -lgmp

generate: main_generator main_generator_10
	$(BUILD_DIR)/$(TARGET_MNG) all
	$(BUILD_DIR)/$(TARGET_MNG_10) all

pols_generator: $(BUILD_DIR)/$(TARGET_PLG)

$(BUILD_DIR)/$(TARGET_PLG): ./src/pols_generator/pols_generator.cpp ./src/config/definitions.hpp
	$(MKDIR_P) $(BUILD_DIR)
	g++ -g ./src/pols_generator/pols_generator.cpp -o $@ -lgmp

pols: pols_generator
	$(BUILD_DIR)/$(TARGET_PLG)

pols_diff: $(BUILD_DIR)/$(TARGET_PLD)

$(BUILD_DIR)/$(TARGET_PLD): ./src/pols_diff/pols_diff.cpp 
	$(MKDIR_P) $(BUILD_DIR)
	g++ -g ./src/pols_diff/pols_diff.cpp ./src/config/fork_info.* $(CXXFLAGS) $(INC_FLAGS) -o $@ $(LDFLAGS) 

witness2db: $(BUILD_DIR)/$(TARGET_W2DB)

$(BUILD_DIR)/$(TARGET_W2DB): $(OBJS_W2DB)
	$(CXX) $(OBJS_W2DB) $(CXXFLAGS_W2DB) -o $@ $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS_W2DB) $(LDFLAGS_W2DB)

fflonk_setup: $(BUILD_DIR)/$(TARGET_SETUP)

$(BUILD_DIR)/$(TARGET_SETUP): $(OBJS_SETUP)
	$(CXX) $(OBJS_SETUP) $(CXXFLAGS) $(LDFLAGS) -o $@

.PHONY: clean

clean:
	$(RM) -rf $(BUILD_DIR)
	$(RM) -rf $(BUILD_DIR_GPU)
	find . -name main_exec_generated*pp -delete

-include $(DEPS_ZKP)
-include $(DEPS_SETUP)
-include $(DEPS_BCT)

MKDIR_P ?= mkdir -p

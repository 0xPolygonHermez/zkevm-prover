TARGET_ZKP := zkProver
TARGET_BCT := bctree
TARGET_MNG += mainGenerator
TARGET_PLG += polsGenerator
TARGET_PLD += polsDiff
TARGET_TEST := zkProverTest

BUILD_DIR := ./build
SRC_DIRS := ./src ./test ./tools

GRPCPP_FLAGS := $(shell pkg-config grpc++ --cflags)
GRPCPP_LIBS := $(shell pkg-config grpc++ --libs) -lgrpc++_reflection
ifndef GRPCPP_LIBS
$(error gRPC++ could not be found via pkg-config, you need to install them)
endif

CXX := g++
AS := nasm
CXXFLAGS := -std=c++17 $(GRPCPP_FLAGS) #-Wfatal-errors
LDFLAGS := -lprotobuf -lsodium -lgpr -lpthread -lpqxx -lpq -lgmp -lstdc++ -lgmpxx -lsecp256k1 -lcrypto -luuid -liomp5 $(GRPCPP_LIBS)
CFLAGS := -fopenmp -Wall -pthread -Wno-unused-label -fopenmp -mavx2
ASFLAGS := -felf64

# Debug build flags
ifeq ($(dbg),1)
      CFLAGS += -g -D DEBUG
else
      CFLAGS += -O3 -flto=auto -fno-fat-lto-objects

      # Optimizing to run locally. A distributable binary
      # should use a more general setting.
      $(info "Optimizing to run locally.")
      $(info "The binary might not work in a processor with a different set of extensions.")
      CFLAGS += -march=native -mtune=native
endif


# Verify if AVX-512 is supported
# for now disabled, to enable it, you only need to uncomment these lines
#AVX512_SUPPORTED := $(shell cat /proc/cpuinfo | grep -E 'avx512' -m 1)

#ifneq ($(AVX512_SUPPORTED),)
#	CFLAGS += -mavx512f -D__AVX512__
#endif

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP

PROTOC = protoc
PROTOS_PATH = ./src/grpc

PROTO_SRCS := $(shell find $(PROTOS_PATH) -name *.proto)
PROTO_CC   := $(PROTO_SRCS:%.proto=$(BUILD_DIR)/gen/%.pb.cc) $(PROTO_SRCS:%.proto=$(BUILD_DIR)/gen/%.grpc.pb.cc)
PROTO_H    := $(PROTO_SRCS:%.proto=$(BUILD_DIR)/gen/%.pb.h) $(PROTO_SRCS:%.proto=$(BUILD_DIR)/gen/%.grpc.pb.h)
PROTO_OBJS := $(PROTO_CC:%=$(BUILD_DIR)/%.o)

GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

INC_DIRS := $(shell find $(SRC_DIRS) -type d) $(sort $(dir))
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

SRCS_ZKP := $(shell find $(SRC_DIRS) ! -path "./tools/starkpil/bctree/*" ! -path "./test/examples/*" ! -path "./test/prover/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/tests/*" ! -path "./src/main_generator/*" ! -path "./src/pols_generator/*" ! -path "./src/pols_diff/*" -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_ZKP := $(SRCS_ZKP:%=$(BUILD_DIR)/%.o)
DEPS_ZKP := $(OBJS_ZKP:.o=.d)

SRCS_BCT := ./tools/starkpil/bctree/build_const_tree.cpp ./tools/starkpil/bctree/main.cpp ./src/goldilocks/src/goldilocks_base_field.cpp ./src/ffiasm/fr.cpp ./src/ffiasm/fr.asm ./src/starkpil/merkleTree/merkleTreeBN128.cpp ./src/poseidon_opt/poseidon_opt.cpp ./src/goldilocks/src/poseidon_goldilocks.cpp
OBJS_BCT := $(SRCS_BCT:%=$(BUILD_DIR)/%.o) $(PROTO_OBJS)
DEPS_BCT := $(OBJS_BCT:.o=.d)

SRCS_TEST := $(shell find ./test/examples/ ./src/XKCP ./src/goldilocks/src ./src/starkpil/stark_info.* ./src/starkpil/starks.* ./src/starkpil/chelpers.* ./src/rapidsnark/binfile_utils.* ./src/starkpil/steps.* ./src/starkpil/polinomial.hpp ./src/starkpil/merkleTree/merkleTreeGL.* ./src/starkpil/transcript/transcript.* ./src/starkpil/fri ./src/ffiasm ./src/utils ! -path "./src/starkpil/fri/friProveC12.*" -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_TEST := $(SRCS_TEST:%=$(BUILD_DIR)/%.o) $(PROTO_OBJS)
DEPS_TEST := $(OBJS_TEST:.o=.d)

all: $(BUILD_DIR)/$(TARGET_ZKP)

bctree: $(BUILD_DIR)/$(TARGET_BCT)

test: $(BUILD_DIR)/$(TARGET_TEST)

$(BUILD_DIR)/$(TARGET_ZKP): $(OBJS_ZKP)
	$(CXX) $(OBJS_ZKP) -o $@ $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)

$(BUILD_DIR)/$(TARGET_BCT): $(OBJS_BCT)
	$(CXX) $(OBJS_BCT) -o $@ $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)

$(BUILD_DIR)/$(TARGET_TEST): $(OBJS_TEST)
	$(CXX) $(OBJS_TEST) -o $@ $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)

# protobuf
$(BUILD_DIR)/gen/%.pb.cc $(BUILD_DIR)/gen/%.pb.h $(BUILD_DIR)/gen/%.grpc.pb.cc $(BUILD_DIR)/gen/%.grpc.pb.h &: %.proto
	$(MKDIR_P) $(dir $@)
	$(PROTOC) -I $(PROTOS_PATH) --grpc_out=$(dir $@) --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH) $<
	$(PROTOC) -I $(PROTOS_PATH) --cpp_out=$(dir $@) $<

# Tells that objects depends on protobuf headers (it would be nice to be more
# fine grained than this, but it would require listing object files
# individually).
$(OBJS_ZKP) $(OBJS_BCT) $(OBJS_TEST): | $(PROTO_H)

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

pols: pols_generator
	$(BUILD_DIR)/$(TARGET_PLG)

pols_diff: $(BUILD_DIR)/$(TARGET_PLD)

$(BUILD_DIR)/$(TARGET_PLD): ./src/pols_diff/pols_diff.cpp
	$(MKDIR_P) $(BUILD_DIR)
	g++ -g ./src/pols_diff/pols_diff.cpp $(CXXFLAGS) $(INC_FLAGS) -o $@ $(LDFLAGS) 

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)
	find . -name main_exec_generated*pp -delete

-include $(DEPS_ZKP)
-include $(DEPS_BCT)

MKDIR_P ?= mkdir -p

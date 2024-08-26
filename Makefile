TARGET_ZKP := zkProver
TARGET_BCT := bctree
TARGET_ZKEVM_LIB := libzkevm.a
TARGET_STARKS_LIB := libstarks.a
TARGET_MNG += mainGenerator
TARGET_PLG += polsGenerator
TARGET_PLD += polsDiff
TARGET_TEST := zkProverTest
TARGET_SETUP := fflonkSetup
TARGET_CONSTRAINT := constraintChecker

BUILD_DIR := ./build
LIB_DIR := ./lib
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

# Verify if AVX-512 is supported
# for now disabled, to enable it, you only need to uncomment these lines
#AVX512_SUPPORTED := $(shell cat /proc/cpuinfo | grep -E 'avx512' -m 1)

#ifneq ($(AVX512_SUPPORTED),)
#	CXXFLAGS += -mavx512f -D__AVX512__
#endif

# Debug build flags
ifeq ($(dbg),1)
      CXXFLAGS += -g -D__DEBUG__
else
      CXXFLAGS += -O3
endif

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) $(INC_FLAGS_EXT) -MMD -MP

GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

INC_DIRS := $(shell find $(SRC_DIRS) -type d) $(sort $(dir))
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

SRCS_ZKP := $(shell find $(SRC_DIRS) ! -path "./src/constraint_checker/*" ! -path "./src/fflonk_setup/fflonk_setup*" ! -path "./tools/starkpil/bctree/*" ! -path "./test/examples/*" ! -path "./test/prover/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/tests/*" ! -path "./src/main_generator/*" ! -path "./src/pols_generator/*" ! -path "./src/pols_diff/*" -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_ZKP := $(SRCS_ZKP:%=$(BUILD_DIR)/%.o)
DEPS_ZKP := $(OBJS_ZKP:.o=.d)

SRCS_ZKEVM_LIB := $(shell find $(SRC_DIRS) \
	! -path "./src/constraint_checker/*" \
	! -path "./src/main.cpp" \
	! -path "./tools/starkpil/bctree/*" \
	! \( -path "./test/*" -and ! \( -path "./test/service/*" -o -path "./test/utils/*" \) \) \
	! -path "./src/goldilocks/benchs/*" \
	! -path "./src/goldilocks/tests/*" \
	! -path "./src/main_generator/*" \
	! -path "./src/pols_generator/*" \
	! -path "./src/pols_diff/*" \
	! -path "./src/rapidsnark/*" \
	! -path ".src/fflonk_setup/*" \
	-name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_ZKEVM_LIB := $(SRCS_ZKEVM_LIB:%=$(BUILD_DIR)/%.o)
DEPS_ZKEVM_LIB := $(OBJS_ZKEVM_LIB:.o=.d)

SRCS_STARKS_LIB := $(shell find ./src/api/starks_api.* ./src/hint ./src/XKCP ./src/goldilocks/src ./src/config ./src/poseidon_opt/ ./src/starkpil/proof2zkinStark.* ./src/starkpil/stark_info.* ./src/starkpil/starks.* ./src/starkpil/chelpers.* ./src/starkpil/expressions_builder.hpp ./src/rapidsnark/binfile_utils.* ./src/starkpil/steps.* ./src/starkpil/polinomial.hpp ./src/starkpil/merkleTree/* ./src/starkpil/transcript/* ./src/starkpil/fri/* ./src/ffiasm ./src/utils -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_STARKS_LIB := $(SRCS_STARKS_LIB:%=$(BUILD_DIR)/%.o)
DEPS_STARKS_LIB := $(OBJS_STARKS_LIB:.o=.d)

SRCS_BCT := $(shell find ./tools/starkpil/bctree/build_const_tree.cpp ./tools/starkpil/bctree/main.cpp ./src/goldilocks/src ./src/starkpil/merkleTree/merkleTreeBN128.cpp ./src/starkpil/merkleTree/merkleTreeGL.cpp ./src/poseidon_opt/poseidon_opt.cpp ./src/XKCP ./src/ffiasm ./src/utils/* -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_BCT := $(SRCS_BCT:%=$(BUILD_DIR)/%.o)
DEPS_BCT := $(OBJS_BCT:.o=.d)

SRCS_TEST := $(shell find ./test/examples/ ./src/hint ./src/XKCP ./src/goldilocks/src ./src/poseidon_opt/ ./src/starkpil/proof2zkinStark.* ./src/starkpil/stark_info.* ./src/starkpil/starks.* ./src/starkpil/chelpers.* ./src/rapidsnark/binfile_utils.* ./src/starkpil/steps.* ./src/starkpil/polinomial.hpp ./src/starkpil/merkleTree/* ./src/starkpil/transcript/* ./src/starkpil/fri/* ./src/ffiasm ./src/utils -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_TEST := $(SRCS_TEST:%=$(BUILD_DIR)/%.o)
DEPS_TEST := $(OBJS_TEST:.o=.d)

SRCS_CONSTRAINT := $(shell find ./src/constraint_checker ./src/hint ./src/XKCP ./src/goldilocks/src ./src/poseidon_opt/ ./src/starkpil/proof2zkinStark.* ./src/starkpil/stark_info.* ./src/starkpil/starks.* ./src/starkpil/chelpers.* ./src/rapidsnark/binfile_utils.* ./src/starkpil/steps.* ./src/starkpil/polinomial.hpp ./src/starkpil/merkleTree/* ./src/starkpil/transcript/* ./src/starkpil/fri/* ./src/ffiasm ./src/utils -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_CONSTRAINT := $(SRCS_CONSTRAINT:%=$(BUILD_DIR)/%.o)
DEPS_CONSTRAINT := $(OBJS_CONSTRAINT:.o=.d)

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

zkevm_lib: CXXFLAGS_EXT := -D__ZKEVM_LIB__
zkevm_lib: LDFLAGS_EXT  := -L../zkevm-prover-rust/target/release -lzkevm_sm
zkevm_lib: INC_FLAGS_EXT := -I./../zkevm-prover-rust/include
zkevm_lib: $(LIB_DIR)/$(TARGET_ZKEVM_LIB)

starks_lib: CXXFLAGS_EXT := -D__ZKEVM_LIB__ -fPIC#we decided to use the same flags for both libraries
starks_lib: $(LIB_DIR)/$(TARGET_STARKS_LIB)

bctree: $(BUILD_DIR)/$(TARGET_BCT)

fflonk_setup: $(BUILD_DIR)/$(TARGET_SETUP)

test: $(BUILD_DIR)/$(TARGET_TEST)

constraint_checker: $(BUILD_DIR)/$(TARGET_CONSTRAINT)

$(LIB_DIR)/$(TARGET_ZKEVM_LIB): $(OBJS_ZKEVM_LIB)
	mkdir -p $(LIB_DIR)
	mkdir -p $(LIB_DIR)/include
	$(AR) rcs $@ $^
	cp src/api/zkevm_api.hpp $(LIB_DIR)/include/zkevm_lib.h

$(LIB_DIR)/$(TARGET_STARKS_LIB): $(OBJS_STARKS_LIB)
	mkdir -p $(LIB_DIR)
	mkdir -p $(LIB_DIR)/include
	$(AR) rcs $@ $^
	cp src/api/starks_api.hpp $(LIB_DIR)/include/starks_lib.h

$(BUILD_DIR)/$(TARGET_ZKP): $(OBJS_ZKP)
	$(CXX) $(OBJS_ZKP) $(CXXFLAGS) $(CXXFLAGS_EXT) -o $@ $(LDFLAGS) $(LDFLAGS_EXT) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT)

$(BUILD_DIR)/$(TARGET_BCT): $(OBJS_BCT)
	$(CXX) $(OBJS_BCT) $(CXXFLAGS) $(CXXFLAGS_EXT) -o $@ $(LDFLAGS) $(LDFLAGS_EXT) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT)

$(BUILD_DIR)/$(TARGET_TEST): $(OBJS_TEST)
	$(CXX) $(OBJS_TEST) $(CXXFLAGS) $(CXXFLAGS_EXT) -o $@ $(LDFLAGS) $(LDFLAGS_EXT) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT)

$(BUILD_DIR)/$(TARGET_CONSTRAINT): $(OBJS_CONSTRAINT)
	$(CXX) $(OBJS_CONSTRAINT) $(CXXFLAGS) $(CXXFLAGS_EXT) -o $@ $(LDFLAGS) $(LDFLAGS_EXT) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT)

$(BUILD_DIR)/$(TARGET_SETUP): $(OBJS_SETUP)
	$(CXX) $(OBJS_SETUP) $(CXXFLAGS) $(CXXFLAGS_EXT) -o $@ $(LDFLAGS) $(LDFLAGS_EXT) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT)

# assembly
$(BUILD_DIR)/%.asm.o: %.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT) -c $< -o $@

$(BUILD_DIR)/%.cc.o: %.cc
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT) -c $< -o $@

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
	g++ -g ./src/pols_diff/pols_diff.cpp $(CXXFLAGS) $(CXXFLAGS_EXT) $(INC_FLAGS) $(INC_FLAGS_EXT) -o $@ $(LDFLAGS) $(LDFLAGS_EXT)

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)
	find . -name main_exec_generated*pp -delete

-include $(DEPS_ZKP)
-include $(DEPS_SETUP)
-include $(DEPS_BCT)

MKDIR_P ?= mkdir -p

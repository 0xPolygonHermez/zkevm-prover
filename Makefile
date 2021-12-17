TARGET_EXEC := zkProver

BUILD_DIR := ./build
SRC_DIRS := ./src

CXX := g++
CXX := clang++
AS := nasm
CXXFLAGS := -std=c++17 -O3 -Wall
LDFLAGS :=  -lpthread -lgmp -lstdc++ -lomp -lgmpxx -lsecp256k1
CFLAGS := -fopenmp -D'memset_s(W,WL,V,OL)=memset(W,V,OL)'

UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		ASFLAGS := -felf64 
	endif
	ifeq ($(UNAME_S),Darwin)
		ASFLAGS := -fmacho64 --prefix _
	endif

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.asm)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP


$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# assembly
$(BUILD_DIR)/%.asm.o: %.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p
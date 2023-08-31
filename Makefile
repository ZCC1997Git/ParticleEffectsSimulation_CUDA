# Makefile Variables #
# ------------------ #

# The name of the generated executable file
EXECUTABLE=Particles

# Directory names used for incremental build files, executable files, and the executable type (release/debug)
BUILD_DIR := build
BIN_DIR := bin
RELEASE_DIR := release
DEBUG_DIR := debug

# C and CUDA source files to be compiled
# Add any new source files you create to be compiled and linked into the executables to the list
SRCS=src/main.cu src/helper.c src/cpu.c src/openmp.c src/cuda.cu 

# Header files which if changed will result in a rebuild.
# Add any additional header files you create to this list for rebuild support.
DEPS=src/common.h src/config.h src/cpu.h src/cuda.cuh src/helper.h src/main.h src/openmp.h external/stb_image_write.h

# Generate the list of object files from the list of source files.
OBJS=$(addsuffix .o,$(SRCS))

# Select the host C compiler and provide compiler options, for all builds, release builds and debug builds.
CC=gcc
CCFLAGS= -fopenmp -I . -I src -Wall
CCFLAGS_RELEASE= -O3 -DNDEBUG
# -O1 is passed to gcc for debug builds to allow linking via nvcc with inline methods in a C host compiler. This prevents debugging of the inline methods unfortunately
CCFLAGS_DEBUG= -g -O1 -DDEBUG

NVCC=nvcc
NVCCFLAGS= $(GENCODE_FLAGS) -I. -Isrc 
NVCCFLAGS_RELEASE= -lineinfo -O3 -DNDEBUG
NVCCFLAGS_DEBUG= -g -G -DDEBUG

# Build rules #
# ----------- #

# Default build rule - build the release executable
all: release

# Build the release mode executbale (and ensure directories exist)
release: $(BIN_DIR)/$(RELEASE_DIR)/$(EXECUTABLE)

# Build the debug mode executabale (and ensure directories exist)
debug: $(BIN_DIR)/$(DEBUG_DIR)/$(EXECUTABLE)

# Makefile rules using special Makefile variables:
#   $@ is left hand side of rule
#   $< is first item from the right hand side of rule
#   $^ is all items from right hand side of the rule

# Compile CUDA object files for release builds
$(BUILD_DIR)/$(RELEASE_DIR)/%.cu.o : %.cu $(DEPS) $(MAKEFILE_LIST)
	@mkdir -p $(dir $@)
	$(NVCC) -c -o $@ $< $(NVCCFLAGS) $(NVCCFLAGS_RELEASE) 
# Compiler C object files for release builds
$(BUILD_DIR)/$(RELEASE_DIR)/%.c.o : %.c $(DEPS) $(MAKEFILE_LIST)
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CCFLAGS) $(CCFLAGS_RELEASE) 
# Link the executable for release builds
$(BIN_DIR)/$(RELEASE_DIR)/$(EXECUTABLE) : $(addprefix $(BUILD_DIR)/$(RELEASE_DIR)/,$(OBJS))
	@mkdir -p $(dir $@)
	$(NVCC) -o $@ $^ $(NVCCFLAGS) $(NVCCFLAGS_RELEASE) -Xcompiler -fopenmp

# Rules for debug objectss / executbales. Note that these are duplicates with minor changes.
$(BUILD_DIR)/$(DEBUG_DIR)/%.cu.o : %.cu $(DEPS) $(MAKEFILE_LIST)
	@mkdir -p $(dir $@)
	$(NVCC) -c -o $@ $< $(NVCCFLAGS) $(NVCCFLAGS_DEBUG) 
# Compiler C object files for debug builds
$(BUILD_DIR)/$(DEBUG_DIR)/%.c.o : %.c $(DEPS) $(MAKEFILE_LIST)
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CCFLAGS) $(CCFLAGS_DEBUG)
# Link the executable for debug builds
$(BIN_DIR)/$(DEBUG_DIR)/$(EXECUTABLE) : $(addprefix $(BUILD_DIR)/$(DEBUG_DIR)/,$(OBJS))
	@mkdir -p $(dir $@)
	$(NVCC) -o $@ $^ $(NVCCFLAGS) $(NVCCFLAGS_DEBUG)  -Xcompiler -fopenmp


# PHONY rules do not generate files with the same name as the rule.
.PHONY : all release debug clean help
# Clean generated files.
clean:
	@echo "clean"
	@rm -rf $(BIN_DIR) 
	@rm -rf $(BUILD_DIR) 
# Provide usage instructions
help: 
	@echo " Usage:"
	@echo "   make help       Shows this help documentation"
	@echo "   make all        Build the default configuraiton (release)"
	@echo "   make release    Build the release executable ($(BIN_DIR)/$(RELEASE_DIR)/$(EXECUTABLE)) "
	@echo "   make debug      Build the release executable ($(BIN_DIR)/$(DEBUG_DIR)/$(EXECUTABLE)) "
	@echo "   make clean      Clean the build and bin directories"
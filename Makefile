# CUDA toolchain path
CUDA_DIR = /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
CC := $(CUDA_DIR)/bin/nvcc

# Target install directory
DESTDIR = /usr/local

# Build flags
CUDA_ARCH_FLAGS := \
    -arch=sm_60 \
    -gencode=arch=compute_60,code=sm_60 \
    -gencode=arch=compute_61,code=sm_61 \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_75,code=sm_75 \
    -gencode=arch=compute_75,code=compute_75
CC_FLAGS += $(CUDA_ARCH_FLAGS) -I. -O3 -Xcompiler -fPIC --default-stream per-thread
LD_FLAGS := -Xcompiler -fPIC -shared
IOFILTER_CFLAGS := $(shell pkg-config --cflags hdf5)
IOFILTER_LDFLAGS := $(shell pkg-config --libs hdf5)

LIB_OBJ = snappy_compress.o snappy_decompress.o
MAIN_OBJ = $(LIB_OBJ) snappy_cuda.o
DEC_OBJ = $(LIB_OBJ) decompress_measure.o
IOFILTER_OBJ = snappy_iofilter.o
LIBUDF_READER_OBJ = gds_interface.o

# Main targets
all: snappy_cuda libsnappy_cuda_iofilter.so libudf_snappy_reader.so decompress_measure

decompress_measure: $(DEC_OBJ)
	$(CC) $^ $(CUDA_ARCH_FLAGS) -o $@

snappy_cuda : $(MAIN_OBJ)
	$(CC) $^ $(CUDA_ARCH_FLAGS) -o $@

libsnappy_cuda_iofilter.so: $(IOFILTER_OBJ) $(LIB_OBJ)
	$(CC) $^ $(CUDA_ARCH_FLAGS) $(IOFILTER_LDFLAGS) $(LD_FLAGS) -o $@

libudf_snappy_reader.so: $(LIBUDF_READER_OBJ) $(LIB_OBJ)
	$(CC) $^ $(CUDA_ARCH_FLAGS) $(IOFILTER_LDFLAGS) $(LD_FLAGS) -o $@

snappy_cuda.o: snappy_cuda.cu snappy_cuda.h
	$(CC) -c  $< $(CC_FLAGS)
decompress_measure.o: decompress.cu snappy_cuda.h
	$(CC) -c  $< $(CC_FLAGS)
snappy_compress.o: snappy_compress.cu snappy_compress.h
	$(CC) -c  $< $(CC_FLAGS)
snappy_decompress.o: snappy_decompress.cu snappy_decompress.h
	$(CC) -c  $< $(CC_FLAGS)
snappy_iofilter.o: snappy_iofilter.cu snappy_iofilter.h
	$(CC) -c  $< $(CC_FLAGS) $(IOFILTER_CFLAGS)
gds_interface.o: gds_interface.cu gds_interface.h
	$(CC) -c  $< $(CC_FLAGS) $(IOFILTER_CFLAGS)

install:
	@install -v -d $(DESTDIR)/bin $(DESTDIR)/lib $(DESTDIR)/include/snappy-cuda
	@install -v snappy_cuda $(DESTDIR)/bin
	@install -v libudf_snappy_reader.so $(DESTDIR)/lib
	@install -v *.h $(DESTDIR)/include/snappy-cuda
	@if [ -e libsnappy_cuda_iofilter.so ]; then \
		install -v -d $(DESTDIR)/hdf5/lib/plugin; \
		install -v libsnappy_cuda_iofilter.so $(DESTDIR)/hdf5/lib/plugin; \
	fi

clean: 
	rm -f snappy_cuda libsnappy_cuda_iofilter.so libudf_snappy_reader.so *.o

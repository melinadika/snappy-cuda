#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <hdf5.h>

#include "snappy_cuda.h"
#include "snappy_compress.h"
#include "snappy_decompress.h"
#include "snappy_iofilter.h"

//
// Calculate the difference between two timeval structs. This symbol
// is needed by snappy_compress() and snappy_decompress().
//
 double get_runtime(struct timeval *start, struct timeval *end) {
	double start_time = start->tv_sec + start->tv_usec / 1000000.0;
	double end_time = end->tv_sec + end->tv_usec / 1000000.0;
	return (end_time - start_time);
}

enum init_state {
    UNINITIALIZED = 0,
    INITIALIZED_FOR_COMPRESSION,
    INITIALIZED_FOR_DECOMPRESSION,
};

// The HDF5 I/O filter API doesn't let one pass a context around subsequent
// calls to the filter. The best we can do is to make the context buffers
// static to prevent excessive memory [de]allocations.

static struct host_buffer_context *input;
static struct host_buffer_context *output;
static struct program_runtime runtime;
static enum init_state init_state = UNINITIALIZED;

// The current HDF5 filter API doesn't provide hooks to both initialize
// and teardown a filter. We workaround that limitation by initializing
// the input/output contexts once and registering a callback with atexit()
// to destroy them  once the calling program exits.

#define init_buffer(ctx, block_size) \
do { \
    if (runtime.using_cuda && ! runtime.reuse_buffers) { \
        checkCudaErrors(cudaMallocManaged(&ctx, sizeof(host_buffer_context))); \
        memset(ctx, 0, sizeof(host_buffer_context)); \
    } else if (! runtime.reuse_buffers) { \
        ctx = (host_buffer_context *) calloc(1, sizeof(host_buffer_context)); \
    } \
    ctx->max = ULONG_MAX; \
    ctx->block_size = block_size; \
} while (0)

static bool init_writer(int block_size, void *buf, size_t buf_size)
{
    init_buffer(input, block_size);
    init_buffer(output, block_size);

    if (runtime.using_cuda) {
        if (buf_size > input->max) {
            fprintf(stderr, "Input buffer size is too big (%ld > %ld)\n",
                buf_size, input->max);
            return false;
        }
        input->length = buf_size;
        input->total_size = ALIGN_LONG(input->length, 8) * sizeof(*(input->buffer));
        if (! runtime.reuse_buffers) {
            // Input buffer is provided by HDF5, but we need to copy that data to
            // GPU memory too.
            checkCudaErrors(cudaMallocManaged(&input->buffer, input->total_size));
        }
        input->curr = input->buffer;

        // Initialize output buffer that will hold the compressed data,
        // i.e.: output->buffer, output->curr, output->length
        setup_compression_cuda(input, output, &runtime);
    } else {
        // Use the input buffer provided by HDF5
        input->length = input->total_size = buf_size;
        input->buffer = (uint8_t *) buf;
        input->curr = input->buffer;

        // Initialize output buffer
        setup_compression(input, output, &runtime);
    }
    return true;
}

static bool init_reader(int block_size, void *buf, size_t buf_size)
{
    init_buffer(input, block_size);
    init_buffer(output, block_size);

    if (runtime.using_cuda) {
        // Initialize input buffer
        if (buf_size > input->max) {
            fprintf(stderr, "Input buffer size is too big (%ld > %ld)\n",
                buf_size, input->max);
            return false;
        }

        uint32_t total_size = ALIGN_LONG(buf_size, 8) * sizeof(*(input->buffer));

        if (runtime.reuse_buffers && input->total_size < total_size) {
            // cached input buffer is not large enough to hold new data
            checkCudaErrors(cudaFree(input->buffer));
            checkCudaErrors(cudaMallocManaged(&input->buffer, total_size));
            input->total_size = total_size;
        } else if (! runtime.reuse_buffers) {
            checkCudaErrors(cudaMallocManaged(&input->buffer, total_size));
            input->total_size = total_size;
        }
        input->length = buf_size;
        input->curr = input->buffer;
        memcpy(input->buffer, buf, buf_size);

        // Initialize output buffer (output->buffer, output->curr, output->length)
        setup_decompression_cuda(input, output, &runtime);
    } else {
        // Initialize input buffer
        input->length = input->total_size = buf_size;
        input->buffer = (uint8_t *) buf;
        input->curr = input->buffer;

        // Initialize output buffer
        setup_decompression(input, output, &runtime);
    }
    return true;
}

static void atexit_callback()
{
    if (input && output) {
        terminate_compression(input, output, &runtime);
        terminate_decompression(input, output, &runtime);
        if (runtime.using_cuda) {
            checkCudaErrors(cudaFree(input));
            checkCudaErrors(cudaFree(output));
        } else {
            free(input);
            free(output);
        }
        input = output = NULL;
    }
}

static size_t filter_callback(unsigned int flags, size_t cd_nelmts,
    const unsigned int *cd_values, size_t nbytes, size_t *buf_size, void **buf)
{
    memset(&runtime, 0, sizeof(runtime));
    char *use_cuda = getenv("SNAPPY_USE_CUDA");
    runtime.using_cuda = (use_cuda && strcmp(use_cuda, "0") == 0) ? false : true;

    char *user_block_size = getenv("SNAPPY_BLOCKSIZE");
    int block_size = user_block_size ? atoi(user_block_size) : 32 * 1024;

    if (flags & H5Z_FLAG_REVERSE) {
        // Read path. Here we call the decompressor and update the output buffer
        if (init_state == UNINITIALIZED) {
            if (init_reader(block_size, *buf, *buf_size) == false)
                return 0;
            atexit(atexit_callback);
        } else if (init_state == INITIALIZED_FOR_COMPRESSION) {
            atexit_callback();
            if (init_reader(block_size, *buf, *buf_size) == false)
                return 0;
        } else if (init_state == INITIALIZED_FOR_DECOMPRESSION) {
            runtime.reuse_buffers = true;
            if (init_reader(block_size, *buf, *buf_size) == false)
                return 0;
        }

        init_state = INITIALIZED_FOR_DECOMPRESSION;

        if (runtime.using_cuda) {
            if (snappy_decompress_cuda(input, output, &runtime) != SNAPPY_OK) {
                fprintf(stderr, "Failed to decompress input data on the GPU\n");
                return 0;
            }

            if (*buf_size < output->length) {
                // The buffer provided by HDF5 is not large enough to hold the
                // decompressed data
                char *newbuf = (char *) malloc(sizeof(char) * output->length);
                if (! newbuf) {
                    fprintf(stderr, "Not enough memory to hold the decompressed data\n");
                    return 0;
                }
                free(*buf);
                *buf = newbuf;
            }

            memcpy(*buf, output->buffer, output->length);
            *buf_size = output->length;
        } else {
            if (snappy_decompress_host(input, output) != SNAPPY_OK) {
                fprintf(stderr, "Failed to decompress input data\n");
                return 0;
            }
            if (*buf_size < output->length) {
                // The buffer provided by HDF5 is not large enough to hold the
                // decompressed data
                char *newbuf = (char *) malloc(sizeof(char) * output->length);
                if (! newbuf) {
                    fprintf(stderr, "Not enough memory to hold the decompressed data\n");
                    return 0;
                }
                free(*buf);
                *buf = newbuf;
            }
            memcpy(*buf, output->buffer, output->length);
            *buf_size = output->length;
        }

    } else {
        // Write path. Here we call the compressor and update the output buffer
        if (init_state == UNINITIALIZED) {
            if (init_writer(block_size, *buf, *buf_size) == false)
                return 0;
            atexit(atexit_callback);
        } else if (init_state == INITIALIZED_FOR_DECOMPRESSION) {
            atexit_callback();
            if (init_writer(block_size, *buf, *buf_size) == false)
                return 0;
        } else if (init_state == INITIALIZED_FOR_COMPRESSION) {
            runtime.reuse_buffers = true;
            if (init_writer(block_size, *buf, *buf_size) == false)
                return 0;
        }
        init_state = INITIALIZED_FOR_COMPRESSION;

        if (runtime.using_cuda) {
            memcpy(input->buffer, *buf, input->length);
            if (snappy_compress_cuda(input, output, block_size, &runtime) != SNAPPY_OK) {
                fprintf(stderr, "Failed to compress input data on the GPU\n");
                return 0;
            }

            // Because HDF5 manages the output buffer memory we have no option other
            // than creating a copy of the compressed data; HDF5 doesn't know about
            // cudaMalloc() nor cudaFree().
            if (*buf_size < output->length)
            {
                char *newbuf = (char *) malloc(sizeof(char) * output->length);
                if (! newbuf) {
                    fprintf(stderr, "Not enough memory to hold the compressed data\n");
                    return 0;
                }
                free(*buf);
                *buf = newbuf;
                *buf_size = output->length;
            }
            memcpy(*buf, output->buffer, output->length);
        } else {
            if (snappy_compress_host(input, output, block_size) != SNAPPY_OK) {
                fprintf(stderr, "Failed to compress input data\n");
                return 0;
            }
            memcpy(*buf, output->buffer, output->length);
        }
    }

    return output->length;
}

extern "C" const H5Z_class2_t SNAPPY_CUDA_FILTER[1] = {{
    H5Z_CLASS_T_VERS,
    SNAPPY_CUDA_FILTER_ID,
    1, 1,
    "snappy_cuda_filter",
    NULL, /* can_apply */
    NULL, /* set_local */
    filter_callback,
}};

extern "C" H5PL_type_t H5PLget_plugin_type(void) { return H5PL_TYPE_FILTER; }
extern "C" const void *H5PLget_plugin_info(void) { return SNAPPY_CUDA_FILTER; }

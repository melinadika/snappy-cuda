#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <hdf5.h>

#include "snappy_cuda.h"
#include "snappy_compress.h"
#include "snappy_decompress.h"

//
// Calculate the difference between two timeval structs. This symbol
// is needed by snappy_compress() and snappy_decompress().
//
 double get_runtime(struct timeval *start, struct timeval *end) {
	double start_time = start->tv_sec + start->tv_usec / 1000000.0;
	double end_time = end->tv_sec + end->tv_usec / 1000000.0;
	return (end_time - start_time);
}

struct gds_context {
    struct host_buffer_context *input;
    struct host_buffer_context *output;
    struct program_runtime runtime;
};

void *decompressor_init(
    void *gpu_input_buf, size_t gpu_input_size, size_t gpu_total_input_size,
    void *gpu_output_buf, size_t gpu_output_size)
{
    // Alloc buffers
    auto ctx = new struct gds_context;
    checkCudaErrors(cudaMallocManaged(&ctx->input, sizeof(host_buffer_context)));
    checkCudaErrors(cudaMallocManaged(&ctx->output, sizeof(host_buffer_context)));
    memset(ctx->input, 0, sizeof(host_buffer_context));
    memset(ctx->output, 0, sizeof(host_buffer_context));
    memset(&ctx->runtime, 0, sizeof(struct program_runtime));

    // Setup input
    ctx->input->max = ULONG_MAX;
    ctx->input->buffer = ctx->input->curr = (uint8_t *) gpu_input_buf;
    ctx->input->length = gpu_input_size;
    ctx->input->total_size = gpu_total_input_size;
    
    // Setup output
    ctx->output->max = gpu_output_size;
    ctx->output->buffer = ctx->output->curr = (uint8_t *) gpu_output_buf;

    // Configure runtime
    ctx->runtime.reuse_buffers = true;
    ctx->runtime.using_cuda = true;

    setup_decompression_cuda(ctx->input, ctx->output, &ctx->runtime);
    return (void *) ctx;
}

void decompressor_destroy(void *reader_ctx)
{
    struct gds_context *ctx = (struct gds_context *) reader_ctx;
    
    // Don't let the library free the input/output buffers, as we don't manage them
    ctx->input->buffer = ctx->output->buffer = NULL;

    terminate_decompression(ctx->input, ctx->output, &ctx->runtime);
    checkCudaErrors(cudaFree(ctx->input));
    checkCudaErrors(cudaFree(ctx->output));
    delete ctx;
}

bool decompressor_run(void *reader_ctx)
{
    struct gds_context *ctx = (struct gds_context *) reader_ctx;
    if (snappy_decompress_cuda(ctx->input, ctx->output, &ctx->runtime) != SNAPPY_OK) {
        fprintf(stderr, "Failed to decompress input data on the GPU\n");
        return false;
    }
    return true;
}
#ifndef _SNAPPY_GDS_INTERFACE_H_
#define _SNAPPY_GDS_INTERFACE_H_

#include "snappy_cuda.h"

/*
 * Initialize the decompression memory buffers that will be reused across
 * multiple calls to the decompressor. The caller is expected to provide a
 * large enough buffer for the output data. No safety checks are made.
 *
 * @param gpu_input_buf buffer holding compressed data
 * @param gpu_input_size size of gpu_input_buf, in bytes
 * @param gpu_output_buf buffer where decompressed data will be saved
 * @param gpu_output_size size of gpu_output_buf, in bytes
 * @return a pointer to the newly allocated context on success or NULL on failure.
 */
void *decompressor_init(
    void *gpu_input_buf, size_t gpu_input_size, size_t gpu_total_input_size,
    void *gpu_output_buf, size_t gpu_output_size);

/*
 * Deallocate memory buffers previously allocated by decompressor_init().
 *
 * @param reader_ctx decompression context produced by decompressor_init()
 */
void decompressor_destroy(void *reader_ctx);

/* Executes the decompressor, populating the output buffer previously provided
 * to decompressor_init() with the decompressed data.
 *
 * @param reader_ctx decompression context produced by decompressor_init()
 * @return true on success or false on error.
 */
bool decompressor_run(void *reader_ctx);

#endif /* _SNAPPY_GDS_INTERFACE_H */

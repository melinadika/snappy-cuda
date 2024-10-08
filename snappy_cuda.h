#ifndef _SNAPPY_CUDA_H_
#define _SNAPPY_CUDA_H_

#include "common.h"
#include <sys/time.h>

#include <helper_cuda.h>

#define GET_ELEMENT_TYPE(_tag)  (_tag & BITMASK(2))
#define GET_LENGTH_1_BYTE(_tag) ((_tag >> 2) & BITMASK(3))
#define GET_OFFSET_1_BYTE(_tag) ((_tag >> 5) & BITMASK(3))
#define GET_LENGTH_2_BYTE(_tag) ((_tag >> 2) & BITMASK(6))

#define ALIGN_LONG(_p, _width) (((long)_p + (_width-1)) & (0-_width))

// Max length of the input and output files
#define MAX_FILE_LENGTH MEGABYTE(30)

// Return values
typedef enum {
	SNAPPY_OK = 0,				// Success code
	SNAPPY_INVALID_INPUT,		// Input file has an invalid format
	SNAPPY_BUFFER_TOO_SMALL		// Input or output file size is too large
} snappy_status;

// Snappy tag types
enum element_type
{
	EL_TYPE_LITERAL,
	EL_TYPE_COPY_1,
	EL_TYPE_COPY_2,
	EL_TYPE_COPY_4
};

// Auxiliary data needed by the compression and decompression codes.
typedef struct compression_aux_t {
	uint32_t *input_block_size_array;
	uint32_t *output_offsets;
	uint16_t **table;
	uint32_t total_blocks;
} compression_aux_t;

typedef struct decompression_aux_t {
	uint8_t **input_currents;
	uint32_t *input_offsets;
	uint32_t total_blocks;
} decompression_aux_t;

// Buffer context struct for input and output buffers on host
typedef struct host_buffer_context
{
	const char *file_name;		// File name
	uint8_t *buffer;		// Entire buffer
	uint8_t *curr;			// Pointer to current location in buffer
	unsigned long length;		// Length of buffer
	unsigned long max;		// Maximum allowed lenght of buffer
	uint32_t block_size;		// 32K default. This is used in CUDA code
	uint32_t total_size;		// total allocated size, used for cudamemprefetchasynch
	compression_aux_t compression_aux;
	decompression_aux_t decompression_aux;
} host_buffer_context;

// Breakdown of time spent doing each action
struct program_runtime {
	double pre;
	double d_alloc;
	double load;	
	double copy_in;
	double run;
	double copy_out;
	double d_free;
	int blocks;
	int threads_per_block;
	bool reuse_buffers;
	bool using_cuda;
};

/**
 * Calculate the difference between two timeval structs.
 */
double get_runtime(struct timeval *start, struct timeval *end);

#endif	/* _SNAPPY_CUDA_H_ */


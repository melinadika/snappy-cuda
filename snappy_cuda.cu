#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <getopt.h>
#include <string>
#include <iostream>

#include "snappy_cuda.h"
#include "snappy_compress.h"
#include "snappy_decompress.h"


const char options[]="dcb:i:o:x:y:";

/**
 * Read the contents of a file into an in-memory buffer. Upon success,
 * writes the amount read to input->length.
 *
 * @param in_file: input file name.
 * @param input: holds input buffer information
 * @return 1 if file does not exist, is too long, or different number of bytes
 *         were read than expected, 0 otherwise
 */
static int read_input_host(char *in_file, struct host_buffer_context *input)
{
	FILE *fin = fopen(in_file, "r");
	if (fin == NULL) {
		fprintf(stderr, "Invalid input file: %s\n", in_file);
		return 1;
	}

	fseek(fin, 0, SEEK_END);
	input->length = ftell(fin);
	fseek(fin, 0, SEEK_SET);

	if (input->length > input->max) {
		fprintf(stderr, "input_size is too big (%ld > %ld)\n",
				input->length, input->max);
		return 1;
	}

	input->buffer = (uint8_t *)malloc(ALIGN_LONG(input->length, 8) * sizeof(*(input->buffer)));
	input->curr = input->buffer;
	size_t n = fread(input->buffer, sizeof(*(input->buffer)), input->length, fin);
	fclose(fin);

#ifdef DEBUG
	printf("%s: read %ld bytes from %s (%lu)\n", __func__, input->length, in_file, n);
#endif
   return (n != input->length);
}

/**
 * Read the contents of a file into an in-memory buffer. Upon success,
 * writes the amount read to input->length.
 *
 * @param in_file: input file name.
 * @param input: holds input buffer information
 * @return 1 if file does not exist, is too long, or different number of bytes
 *         were read than expected, 0 otherwise
 */
static int read_input_cuda(char *in_file, struct host_buffer_context *input)
{
	FILE *fin = fopen(in_file, "r");
	if (fin == NULL) {
		fprintf(stderr, "Invalid input file: %s\n", in_file);
		return 1;
	}

	fseek(fin, 0, SEEK_END);
	input->length = ftell(fin);
	fseek(fin, 0, SEEK_SET);

	if (input->length > input->max) {
		fprintf(stderr, "input_size is too big (%ld > %ld)\n",
				input->length, input->max);
		return 1;
	}

	//input->buffer = (uint8_t *)malloc(ALIGN_LONG(input->length, 8) * sizeof(*(input->buffer)));
	input->total_size = ALIGN_LONG(input->length, 8) * sizeof(*(input->buffer));
	checkCudaErrors(cudaMallocManaged(&input->buffer,input->total_size));
	input->curr = input->buffer;
	size_t n = fread(input->buffer, sizeof(*(input->buffer)), input->length, fin);
	fclose(fin);

#ifdef DEBUG
	printf("%s: read %ld bytes from %s (%lu)\n", __func__, input->length, in_file, n);
#endif
   return (n != input->length);
}

/**
 * Write the contents of the output buffer to a file.
 *
 * @param out_file: output filename.
 * @param output: holds output buffer information
 */
static void write_output_host(char *out_file, struct host_buffer_context *output)
{
	FILE *fout = fopen(out_file, "w");
	fwrite(output->buffer, 1, output->length, fout);
	fclose(fout);
}

/**
 * Print out application usage.
 *
 * @param exe_name: name of the application
 */
static void usage(const char *exe_name)
{
#ifdef DEBUG
	fprintf(stderr, "**DEBUG BUILD**\n");
#endif //DEBUG
	fprintf(stderr, "Compress or decompress a file with Snappy\nCan use either the host CPU or CUDA\n");
	fprintf(stderr, "usage: %s [-d] [-x <cuda blocks>] [-y <cuda threads per block] [-c] [-b <block_size>] -i <input_file> [-o <output_file>]\n", exe_name);
	fprintf(stderr, "d: use CUDA, by default host is used\n");
	fprintf(stderr, "x: Grid size - number of blocks (Carefull! no error checks are done)\n");
	fprintf(stderr, "y: number of threads per block (Carefull! no error checks are done)\n");
	fprintf(stderr, "c: perform compression, by default performs decompression\n");
	fprintf(stderr, "b: block size used for compression, default is 32KB, ignored for decompression\n");
	fprintf(stderr, "i: input file\n");
	fprintf(stderr, "o: output file\n");
}

/**
 * Calculate the difference between two timeval structs.
 */
double get_runtime(struct timeval *start, struct timeval *end) {
	double start_time = start->tv_sec + start->tv_usec / 1000000.0;
	double end_time = end->tv_sec + end->tv_usec / 1000000.0;
	return (end_time - start_time);
}

int main(int argc, char **argv)
{
	int opt;
	snappy_status status;
	
	int compress = 0;
	int block_size = 32 * 1024; // Default is 32KB
    char * input_file = NULL;
    char * output_file = NULL;
    const char * default_output_file = "output.txt";
	struct host_buffer_context *input;
	struct host_buffer_context *output;
	struct program_runtime runtime;

	// use defaults which are set later
	memset(&runtime, 0, sizeof(runtime));

	while ((opt = getopt(argc, argv, options)) != -1)
	{
		switch(opt)
		{
		case 'd':
			runtime.using_cuda = true;
			break;

		case 'c':
			compress = 1;
			break;
		
		case 'b':
			block_size = atoi(optarg);
			break;

		case 'i':
			input_file = optarg;
			break;

		case 'o':
			output_file = optarg;
			break;

		case 'x':
			runtime.blocks = atoi(optarg);
			break;
		
		case 'y':
			runtime.threads_per_block = atoi(optarg);
			break;

		default:
			usage(argv[0]);
			return -2;
		}
	}

	if(runtime.using_cuda)
	{
		checkCudaErrors(cudaMallocManaged(&input,sizeof(host_buffer_context)));
		checkCudaErrors(cudaMallocManaged(&output,sizeof(host_buffer_context)));
	}
	else
	{
		input = (host_buffer_context *)malloc(sizeof(host_buffer_context));
		output = (host_buffer_context *)malloc(sizeof(host_buffer_context));
	}
	


	input->buffer = NULL;
	input->length = 0;
	input->max = ULONG_MAX;

	output->buffer = NULL;
	output->length = 0;
	output->max = ULONG_MAX;

	if (!input_file)
	{
		usage(argv[0]);
		return -1;
	}
	input->file_name = input_file;
	printf("Using input file %s\n", input_file);

	// If no output file was provided, use a default file
	if (output_file == NULL) {
		output_file = (char *)default_output_file;
	}
	output->file_name = output_file;
	printf("Using output file %s\n", output_file);

	// Read the input file into main memory

	if(runtime.using_cuda) {
		if (read_input_cuda(input_file, input))
			return -1;
		input->block_size = block_size;
	}
	else {
	if (read_input_host(input_file, input))
		return -1;
	}

	
	if (compress) {
		
		if (runtime.using_cuda)
		{
			setup_compression_cuda(input, output, &runtime);


			struct timeval start;
			struct timeval end;

			gettimeofday(&start, NULL);	
			status = snappy_compress_cuda(input, output, block_size, &runtime);
			gettimeofday(&end, NULL);

			runtime.run = get_runtime(&start, &end);
		}
		else
		{
			setup_compression(input, output, &runtime);

			struct timeval start;
			struct timeval end;

			gettimeofday(&start, NULL);	
			status = snappy_compress_host(input, output, block_size);
			gettimeofday(&end, NULL);

			runtime.run = get_runtime(&start, &end);
		}
	}
	else {
	
		if (runtime.using_cuda)
		{
			if (setup_decompression_cuda(input, output, &runtime))
				return -1;

			struct timeval start;
			struct timeval end;

			gettimeofday(&start, NULL);
			status = snappy_decompress_cuda(input, output, &runtime);
			gettimeofday(&end, NULL);

			runtime.run = get_runtime(&start, &end);
		}
		else
		{
			if (setup_decompression(input, output, &runtime))
				return -1;

			struct timeval start;
			struct timeval end;

			gettimeofday(&start, NULL);
			status = snappy_decompress_host(input, output);
			gettimeofday(&end, NULL);

			runtime.run = get_runtime(&start, &end);
		}
	}
	
	if (status == SNAPPY_OK)
	{
		// Write the output buffer from main memory to a file
		//if (!(compress && runtime.using_cuda))
			write_output_host(output_file, output);

		if (compress) {
	
			printf("Compressed %ld bytes to: %s\n", output->length, output_file);
			printf("Compression ratio: %f\n", 1 - (double)output->length / (double)input->length);
			
			struct timeval start;
			struct timeval end;
			gettimeofday(&start, NULL);
			terminate_compression(input, output, &runtime);
			gettimeofday(&end, NULL);

			runtime.d_free = get_runtime(&start, &end);
		}
		else {
			printf("Decompressed %ld bytes to: %s\n", output->length, output_file);
			printf("Compression ratio: %f\n", 1 - (double)input->length / (double)output->length);
	
			struct timeval start;
			struct timeval end;
			gettimeofday(&start, NULL);
			terminate_decompression(input, output, &runtime);
			gettimeofday(&end, NULL);

			runtime.d_free = get_runtime(&start, &end);
		}
	
		printf("Pre-processing time: %f\n", runtime.pre);
		printf("Alloc time: %f\n", runtime.d_alloc);
		printf("Load time: %f\n", runtime.load);
		printf("Copy in time: %f\n", runtime.copy_in);
		printf("Elapsed time: %f\n", runtime.run);
		printf("Copy out time: %f\n", runtime.copy_out);
		printf("Free time: %f\n", runtime.d_free);
	}
	else
	{
		fprintf(stderr, "Encountered Snappy error %u\n", status);
		return -1;
	}

	if(runtime.using_cuda)
	{
		checkCudaErrors(cudaFree(input));
		checkCudaErrors(cudaFree(output));
	}
	else
	{
		free(input);
		free(output);
	}
	
	return 0;
}


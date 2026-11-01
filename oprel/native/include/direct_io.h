#ifndef DIRECT_IO_H
#define DIRECT_IO_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define OPREL_SECTOR_SIZE 4096

typedef struct DirectIOFile {
#ifdef _WIN32
    HANDLE handle;
#else
    int fd;
#endif
    char path[1024];
    bool is_open;
    int64_t current_size;
} DirectIOFile;

/**
 * Allocate aligned memory buffer suitable for unbuffered Direct I/O.
 */
void* direct_io_alloc(size_t bytes);

/**
 * Free aligned memory buffer.
 */
void direct_io_free(void* ptr);

/**
 * Open or create file with direct unbuffered I/O flags.
 */
bool direct_io_open(DirectIOFile* f, const char* path, bool create_new);

/**
 * Write aligned buffer to file at specific offset.
 */
bool direct_io_write(DirectIOFile* f, int64_t offset, const void* buffer, size_t bytes);

/**
 * Read from file into aligned buffer at specific offset.
 */
bool direct_io_read(DirectIOFile* f, int64_t offset, void* buffer, size_t bytes);

/**
 * Close direct I/O file handle.
 */
void direct_io_close(DirectIOFile* f);

#ifdef __cplusplus
}
#endif

#endif // DIRECT_IO_H

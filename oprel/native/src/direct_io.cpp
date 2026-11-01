#include "direct_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

void* direct_io_alloc(size_t bytes) {
    if (bytes == 0) return NULL;
    // Align up to sector multiple
    size_t aligned_bytes = (bytes + OPREL_SECTOR_SIZE - 1) & ~(OPREL_SECTOR_SIZE - 1);
#ifdef _WIN32
    void* ptr = VirtualAlloc(NULL, aligned_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    return ptr;
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, OPREL_SECTOR_SIZE, aligned_bytes) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

void direct_io_free(void* ptr) {
    if (!ptr) return;
#ifdef _WIN32
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    free(ptr);
#endif
}

bool direct_io_open(DirectIOFile* f, const char* path, bool create_new) {
    if (!f || !path) return false;
    if (f->is_open) direct_io_close(f);
    memset(f, 0, sizeof(DirectIOFile));
    strncpy(f->path, path, sizeof(f->path) - 1);

#ifdef _WIN32
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    if (wlen <= 0) return false;
    wchar_t* wpath = (wchar_t*)malloc(wlen * sizeof(wchar_t));
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, wlen);

    DWORD creation = create_new ? CREATE_ALWAYS : OPEN_ALWAYS;
    DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH;

    HANDLE h = CreateFileW(
        wpath,
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ,
        NULL,
        creation,
        flags,
        NULL
    );
    free(wpath);

    if (h == INVALID_HANDLE_VALUE) {
        // Fallback without NO_BUFFERING if drive refuses
        flags = FILE_ATTRIBUTE_NORMAL;
        wpath = (wchar_t*)malloc(wlen * sizeof(wchar_t));
        MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, wlen);
        h = CreateFileW(
            wpath,
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ,
            NULL,
            creation,
            flags,
            NULL
        );
        free(wpath);
        if (h == INVALID_HANDLE_VALUE) {
            return false;
        }
    }

    f->handle = h;
    f->is_open = true;

    LARGE_INTEGER sz;
    if (GetFileSizeEx(h, &sz)) {
        f->current_size = sz.QuadPart;
    } else {
        f->current_size = 0;
    }
    return true;
#else
    int flags = O_RDWR | (create_new ? (O_CREAT | O_TRUNC) : O_CREAT);
#ifdef O_DIRECT
    flags |= O_DIRECT;
#endif
    int fd = open(path, flags, 0644);
    if (fd < 0) {
        // fallback without O_DIRECT
        flags &= ~O_DIRECT;
        fd = open(path, flags, 0644);
        if (fd < 0) return false;
    }
    f->fd = fd;
    f->is_open = true;
    f->current_size = lseek(fd, 0, SEEK_END);
    return true;
#endif
}

bool direct_io_write(DirectIOFile* f, int64_t offset, const void* buffer, size_t bytes) {
    if (!f || !f->is_open || !buffer || bytes == 0) return false;
    // Loop: POSIX pwrite may transfer less than requested and Win32 takes a 32-bit length.
    const uint8_t* src = (const uint8_t*)buffer;
    size_t done = 0;
    while (done < bytes) {
#ifdef _WIN32
        LARGE_INTEGER li;
        li.QuadPart = offset + (int64_t)done;
        if (!SetFilePointerEx(f->handle, li, NULL, FILE_BEGIN)) return false;
        DWORD want = (DWORD)((bytes - done) > (size_t)(1u << 30) ? (size_t)(1u << 30) : (bytes - done));
        DWORD written = 0;
        if (!WriteFile(f->handle, src + done, want, &written, NULL) || written == 0) return false;
        done += written;
#else
        ssize_t n = pwrite(f->fd, src + done, bytes - done, offset + (int64_t)done);
        if (n <= 0) return false;
        done += (size_t)n;
#endif
    }
    if (offset + (int64_t)bytes > f->current_size) {
        f->current_size = offset + (int64_t)bytes;
    }
    return true;
}

bool direct_io_read(DirectIOFile* f, int64_t offset, void* buffer, size_t bytes) {
    if (!f || !f->is_open || !buffer || bytes == 0) return false;
    uint8_t* dst = (uint8_t*)buffer;
    size_t done = 0;
    while (done < bytes) {
#ifdef _WIN32
        LARGE_INTEGER li;
        li.QuadPart = offset + (int64_t)done;
        if (!SetFilePointerEx(f->handle, li, NULL, FILE_BEGIN)) return false;
        DWORD want = (DWORD)((bytes - done) > (size_t)(1u << 30) ? (size_t)(1u << 30) : (bytes - done));
        DWORD got = 0;
        if (!ReadFile(f->handle, dst + done, want, &got, NULL) || got == 0) return false;  // 0 = EOF: truncated file
        done += got;
#else
        ssize_t n = pread(f->fd, dst + done, bytes - done, offset + (int64_t)done);
        if (n <= 0) return false;
        done += (size_t)n;
#endif
    }
    return true;
}

void direct_io_close(DirectIOFile* f) {
    if (!f || !f->is_open) return;
#ifdef _WIN32
    CloseHandle(f->handle);
    f->handle = INVALID_HANDLE_VALUE;
#else
    close(f->fd);
    f->fd = -1;
#endif
    f->is_open = false;
}

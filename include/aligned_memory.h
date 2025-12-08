#pragma once

#include <stdlib.h>
#include <memory.h>


#define MEM_DATA_ALIGN 64


//________________________________________________________________________________________________________________________
///
/// \brief Allocate 'size' bytes of uninitialized storage, and return a pointer to the allocated memory block.
///
static inline void* aligned_malloc(size_t size)
{
	#ifdef _WIN32
	return _aligned_malloc(size, MEM_DATA_ALIGN);
	#else
	// round 'size' up to the next multiple of 'MEM_DATA_ALIGN', which must be a power of 2
	return aligned_alloc(MEM_DATA_ALIGN, (size + MEM_DATA_ALIGN - 1) & (-MEM_DATA_ALIGN));
	#endif
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate 'size' bytes of storage initialized with zeros, and return a pointer to the allocated memory block.
///
static inline void* aligned_calloc(size_t size)
{
	void* p = aligned_malloc(size);
	if (p != NULL) {
		memset(p, 0, size);
	}
	return p;
}


//________________________________________________________________________________________________________________________
///
/// \brief Deallocate a previously allocated memory block.
///
static inline void aligned_free(void* memblock)
{
	#ifdef _WIN32
	_aligned_free(memblock);
	#else
	free(memblock);
	#endif
}

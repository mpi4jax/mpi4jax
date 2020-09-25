cdef extern from "cuda_runtime_api.h":
    cdef enum cudaError:
        cudaSuccess

    cdef enum cudaMemcpyKind:
        cudaMemcpyHostToHost
        cudaMemcpyHostToDevice
        cudaMemcpyDeviceToHost
        cudaMemcpyDeviceToDevice
        cudaMemcpyDefault

    ctypedef cudaError cudaError_t

    ctypedef void* cudaStream_t
    cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) nogil

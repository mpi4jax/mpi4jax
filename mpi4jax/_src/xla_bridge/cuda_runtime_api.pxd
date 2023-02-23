cdef extern from "cuda_runtime_api.h":
    cdef enum cudaError:
        cudaSuccess = 0

    cdef enum cudaMemcpyKind:
        cudaMemcpyHostToHost = 0
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2
        cudaMemcpyDeviceToDevice = 3
        cudaMemcpyDefault = 4

    ctypedef cudaError cudaError_t

    ctypedef void* cudaStream_t
    cudaError_t cudaStreamSynchronize(cudaStream_t stream) nogil
    cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) nogil

    const char* cudaGetErrorName(cudaError_t error) nogil
    const char* cudaGetErrorString(cudaError_t error) nogil

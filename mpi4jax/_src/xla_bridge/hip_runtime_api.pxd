cdef extern from "hip/hip_runtime_api.h":
    cdef enum hipError_t:
        hipSuccess = 0

    cdef enum hipMemcpyKind:
        hipMemcpyHostToHost = 0
        hipMemcpyHostToDevice = 1
        hipMemcpyDeviceToHost = 2
        hipMemcpyDeviceToDevice = 3
        hipMemcpyDefault = 4

    ctypedef void* hipStream_t
    hipError_t hipStreamSynchronize(hipStream_t stream) nogil
    hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind) nogil

    const char* hipGetErrorName(hipError_t hip_error) nogil
    const char* hipGetErrorString(hipError_t hip_error) nogil
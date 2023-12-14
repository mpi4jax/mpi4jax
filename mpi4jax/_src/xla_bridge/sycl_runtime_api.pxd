cdef extern from "sycl/CL/sycl.hpp" namespace "cl::sycl":

    cdef cppclass queue "cl::sycl::queue":
            void wait() nogil
            pass


    #cdef enum cudaMemcpyKind:
    #    cudaMemcpyHostToHost = 0
    #    cudaMemcpyHostToDevice = 1
    #    cudaMemcpyDeviceToHost = 2
    #    cudaMemcpyDeviceToDevice = 3
    #    cudaMemcpyDefault = 4

    #ctypedef cudaError cudaError_t

    #cudaError_t cudaStreamSynchronize(cudaStream_t stream) nogil
    #cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) nogil

    #const char* cudaGetErrorName(cudaError_t error) nogil
    #const char* cudaGetErrorString(cudaError_t error) nogil

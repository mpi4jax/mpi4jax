#from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "sycl/CL/sycl.hpp" namespace "cl::sycl":

    cdef cppclass queue "cl::sycl::queue":
        void wait() nogil
        device get_device() nogil
        pass

    cdef cppclass device "cl::sycl::device":
        bool is_accelerator() nogil
        pass

#    cdef enum info_device "cl::sycl::info::device": 
#        name = 58 
#        pass

    #ctypedef cudaError cudaError_t

    #cudaError_t cudaStreamSynchronize(cudaStream_t stream) nogil
    #const char* cudaGetErrorString(cudaError_t error) nogil

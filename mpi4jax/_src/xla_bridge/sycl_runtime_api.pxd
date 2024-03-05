
cdef extern from "sycl/CL/sycl.hpp" namespace "cl::sycl":

    cdef cppclass queue "cl::sycl::queue":
        void wait() except + 
        event memcpy(void*,void*,size_t) except+
        pass

    cdef cppclass event "cl::sycl::event":
        void wait() except + 
        pass
    

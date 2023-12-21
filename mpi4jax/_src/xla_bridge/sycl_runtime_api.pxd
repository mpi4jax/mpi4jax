
cdef extern from "sycl/CL/sycl.hpp" namespace "cl::sycl":

    cdef cppclass queue "cl::sycl::queue":
        void wait() except + 
        pass


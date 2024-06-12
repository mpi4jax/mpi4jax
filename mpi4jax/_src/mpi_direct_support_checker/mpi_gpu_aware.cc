#include <cstdlib>

#include "mpi_gpu_aware.h"

namespace detail {

    static inline int
    EnsureMPIIsInitialized() {
        int is_mpi_initialized = 0;

        if((::MPI_Initialized(&is_mpi_initialized) != MPI_SUCCESS) ||
        (is_mpi_initialized == 0)) {
            return -1;
        }

        // At this point MPI should be initialized. Except if an other thread
        // called MPI_Finalize() in between.

        return 0;
    }

#if defined(MPI_GPU_AWARE_CRAYMPICH_API_SUPPORT)
    static inline int
    getenv() {
        const char* environment_string = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
        if(environment_string == NULL) {
            return -1;
        }

        if(environment_string[0] != '1') {
            return -1;
        }

        return 0;
    }

    /// Rely on MPIX_GPU_query_support. It requires that we have
    /// MPIX_GPU_SUPPORT_CUDA defined to plug into the MPIX_GPU_query_support.
    /// https://www.mpich.org/static/docs/v4.0.x/www3/MPIX_GPU_query_support.html
    ///
    static inline int
    MPIx() {
        int result = 0;

        const auto CheckGPUKindSupport = [](int offset, int gpu_kind) -> int {
            int is_gpu_kind_supported = 0;

            if(::MPIX_GPU_query_support(gpu_kind, &is_gpu_kind_supported) != MPI_SUCCESS) {
                return -1;
            }

            return is_gpu_kind_supported << offset;
        };

    #if defined(MPIX_GPU_SUPPORT_CUDA)
        result |= CheckGPUKindSupport(0, MPIX_GPU_SUPPORT_CUDA);
    #endif

    #if defined(MPIX_GPU_SUPPORT_ZE)
        result |= CheckGPUKindSupport(1, MPIX_GPU_SUPPORT_ZE);
    #endif

    #if defined(MPIX_GPU_SUPPORT_HIP)
        result |= CheckGPUKindSupport(2, MPIX_GPU_SUPPORT_HIP);
    #endif
        return result == 0 ? -1 : result;
    }

    static inline int
    DoCheck() {
        if(getenv() < 0) {
            return -1;
        }

        if(EnsureMPIIsInitialized() < 0) {
            return -1;
        }

        if(MPIx() < 0) {
            return -1;
        }

        return 0;
    }

#elif defined(MPI_GPU_AWARE_OPENMPI_API_SUPPORT)
    static inline int
    MPIx() {
        int result = 0;

    #if defined(MPI_GPU_AWARE_OPENMPI_API_SUPPORT) && defined(OMPI_HAVE_MPI_EXT_CUDA)
        if(::MPIX_Query_cuda_support() != 1) {
            return -1;
        }
        result |= 1 << 0;
    #endif

        // #if defined(MPI_GPU_AWARE_OPENMPI_API_SUPPORT) && defined(OMPI_HAVE_MPI_EXT_ZERO)
        //         if(::MPIX_Query_zero_support() != 1) {
        //             return -1;
        //         }
        //         result |= 1 << 2;
        // #endif

    #if defined(MPI_GPU_AWARE_OPENMPI_API_SUPPORT) && defined(OMPI_HAVE_MPI_EXT_ROCM)
        if(::MPIX_Query_rocm_support() != 1) {
            return -1;
        }
        result |= 1 << 2;
    #endif

        return result == 0 ? -1 : result;
    }

    static inline int
    DoCheck() {
        if(EnsureMPIIsInitialized() < 0) {
            return -1;
        }

        if(MPIx() < 0) {
            return -1;
        }

        return 0;
    }
#elif !defined(MPI_GPU_AWARE_API_SUPPORT)
    static inline int
    DoCheck() {
        return -1;
    }
#endif
} // namespace detail

extern "C" int mpi_gpu_aware(void) {
    static const int result = detail::DoCheck();
    return result;
}
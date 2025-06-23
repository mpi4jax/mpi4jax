#ifndef MPI_GPU_AWARE_H
#define MPI_GPU_AWARE_H

#include <mpi.h>

#if defined(CRAY_MPICH_VERSION) && defined(MPIX_GPU_SUPPORT_CUDA)
    #define MPI_GPU_AWARE_CRAYMPICH_API_SUPPORT       1
    #define MPI_GPU_AWARE_CRAYMPICH_POTENTIAL_SUPPORT 1

#elif defined(OPEN_MPI)

    #include <mpi-ext.h>

    #if(defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM) || \
        (defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA)

        #define MPI_GPU_AWARE_OPENMPI_API_SUPPORT 1

        #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
            #define MPI_GPU_AWARE_OPENMPI_POTENTIAL_SUPPORT 1
        #endif
    #endif
#endif

#if defined(MPI_GPU_AWARE_CRAYMPICH_API_SUPPORT) || \
    defined(MPI_GPU_AWARE_OPENMPI_API_SUPPORT)

    /// The MPI implementation supports a runtime API we can use to ensure GPU
    /// awareness.
    ///
    #define MPI_GPU_AWARE_API_SUPPORT 1
#endif

#if defined(MPI_GPU_AWARE_CRAYMPICH_POTENTIAL_SUPPORT) || \
    defined(MPI_GPU_AWARE_OPENMPI_POTENTIAL_SUPPORT)

    /// We assume it is likely that the MPI implementation is GPU aware ?
    ///
    #define MPI_GPU_AWARE_POTENTIAL_SUPPORT 1
#endif

/// GPU aware MPI is a runtime decision on Cray MPICH. It relies on a library
/// called GPU Transport Layer (GTL). This library is linked by the wrapper when
/// a GPU architecture module is loaded (i.e.: craype-x86-trento).
/// It is not possible to determine at compile time if the MPI implementation
/// will be GPU aware. We can know if the MPI could have that feature enabled,
/// but not if it will be enabled.
/// Returns: < 0 if we could not determine if the MPI is GPU aware.
///
extern "C" int mpi_gpu_aware(void);

#endif
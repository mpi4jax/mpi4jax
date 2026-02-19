// Descriptor structs for passing MPI operation parameters
// These structs use int64_t for all fields to ensure consistent memory layout
// across platforms and compatibility with both OpenMPI and MPICH.
//
// The structs are packed to ensure no padding between fields, making them
// safe to serialize/deserialize between Python (Cython) and C++.

#ifndef MPI4JAX_MPI_DESCRIPTORS_H_
#define MPI4JAX_MPI_DESCRIPTORS_H_

#include <cstdint>

namespace mpi4jax {

#pragma pack(push, 1)

// Sendrecv descriptor - all fields are int64_t for portability
struct SendrecvDescriptor {
    int64_t sendcount;
    int64_t dest;
    int64_t sendtag;
    int64_t sendtype;   // MPI_Datatype handle as int64
    int64_t recvcount;
    int64_t source;
    int64_t recvtag;
    int64_t recvtype;   // MPI_Datatype handle as int64
    int64_t comm;       // MPI_Comm handle as int64
    int64_t status;     // MPI_Status* pointer as int64
};

// Size should be 10 * 8 = 80 bytes
static_assert(sizeof(SendrecvDescriptor) == 80, "SendrecvDescriptor size mismatch");

#pragma pack(pop)

} // namespace mpi4jax

#endif // MPI4JAX_MPI_DESCRIPTORS_H_

#include "mpi_xla_bridge_cpp.hpp"
#include <iostream>

namespace mpi4jax {

// Global debug flag
bool PRINT_DEBUG = false;

void set_logging(bool enable) {
    PRINT_DEBUG = enable;
}

bool get_logging() {
    return PRINT_DEBUG;
}

std::string random_id() {
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, sizeof(alphanum) - 2);

    std::string result;
    result.reserve(8);
    for (int i = 0; i < 8; ++i) {
        result += alphanum[dis(gen)];
    }
    return result;
}

void print_debug(const char* message, const char* rid, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    std::cout << "r" << rank << " | " << rid << " | " << message << std::endl;
    std::cout.flush();
}

int abort_on_error(int ierr, MPI_Comm comm, const char* mpi_op) {
    if (ierr == MPI_SUCCESS) {
        return 0;
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    char error_string[MPI_MAX_ERROR_STRING];
    int length;
    MPI_Error_string(ierr, error_string, &length);

    std::cerr << "r" << rank << " | MPI_" << mpi_op
              << " returned error code " << ierr << ": "
              << std::string(error_string, length) << " - aborting" << std::endl;
    std::cerr.flush();

    return MPI_Abort(comm, ierr);
}

uintptr_t get_mpi_status_ignore_addr() {
    return reinterpret_cast<uintptr_t>(MPI_STATUS_IGNORE);
}

void mpi_sendrecv(
    void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
    void* recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
    MPI_Comm comm, MPI_Status* status
) {
    std::string rid;
    std::chrono::high_resolution_clock::time_point start;

    if (PRINT_DEBUG) {
        rid = random_id();
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "MPI_Sendrecv <- %d (tag %d, %d items) / -> %d (tag %d, %d items)",
                 source, recvtag, recvcount, dest, sendtag, sendcount);
        print_debug(msg, rid.c_str(), comm);
        start = std::chrono::high_resolution_clock::now();
    }

    // MPI Call
    int ierr = MPI_Sendrecv(
        sendbuf, sendcount, sendtype, dest, sendtag,
        recvbuf, recvcount, recvtype, source, recvtag,
        comm, status
    );

    if (PRINT_DEBUG) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1e6;
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Sendrecv done with code %d (%.2es)", ierr, seconds);
        print_debug(msg, rid.c_str(), comm);
    }

    abort_on_error(ierr, comm, "Sendrecv");
}

void mpi_sendrecv_cpu(void** out_ptr, void** data_ptr) {
    // Extract parameters from data_ptr array (same layout as Cython version)
    int sendcount = *reinterpret_cast<int*>(data_ptr[0]);
    void* sendbuf = data_ptr[1];
    int dest = *reinterpret_cast<int*>(data_ptr[2]);
    int sendtag = *reinterpret_cast<int*>(data_ptr[3]);
    MPI_Datatype sendtype = reinterpret_cast<MPI_Datatype>(*reinterpret_cast<uintptr_t*>(data_ptr[4]));

    int recvcount = *reinterpret_cast<int*>(data_ptr[5]);
    int source = *reinterpret_cast<int*>(data_ptr[6]);
    int recvtag = *reinterpret_cast<int*>(data_ptr[7]);
    MPI_Datatype recvtype = reinterpret_cast<MPI_Datatype>(*reinterpret_cast<uintptr_t*>(data_ptr[8]));

    MPI_Comm comm = reinterpret_cast<MPI_Comm>(*reinterpret_cast<uintptr_t*>(data_ptr[9]));
    MPI_Status* status = reinterpret_cast<MPI_Status*>(*reinterpret_cast<uintptr_t*>(data_ptr[10]));

    void* recvbuf = out_ptr[0];

    mpi_sendrecv(
        sendbuf, sendcount, sendtype, dest, sendtag,
        recvbuf, recvcount, recvtype, source, recvtag,
        comm, status
    );
}

} // namespace mpi4jax

from .._src.collective_ops.allgather import mpi_allgather_p
from .._src.collective_ops.allreduce import mpi_allreduce_p
from .._src.collective_ops.alltoall import mpi_alltoall_p
from .._src.collective_ops.barrier import mpi_barrier_p
from .._src.collective_ops.bcast import mpi_bcast_p
from .._src.collective_ops.gather import mpi_gather_p
from .._src.collective_ops.recv import mpi_recv_p
from .._src.collective_ops.reduce import mpi_reduce_p
from .._src.collective_ops.scan import mpi_scan_p
from .._src.collective_ops.send import mpi_send_p
from .._src.collective_ops.scatter import mpi_scatter_p
from .._src.collective_ops.sendrecv import mpi_sendrecv_p


token_override_registry = {}


def mpi_allgather_token_override(in_args, new_token, comm):
    x, _ = in_args
    return mpi_allgather_p.bind(x, new_token, comm=comm)


token_override_registry[mpi_allgather_p] = mpi_allgather_token_override


def mpi_allreduce_token_override(in_args, new_token, op, comm, transpose):
    x, _ = in_args
    return mpi_allreduce_p.bind(x, new_token, op=op, comm=comm, transpose=transpose)


token_override_registry[mpi_allreduce_p] = mpi_allreduce_token_override


def mpi_alltoall_token_override(in_args, new_token, comm):
    x, _ = in_args
    return mpi_alltoall_p.bind(x, new_token, comm=comm)


token_override_registry[mpi_alltoall_p] = mpi_alltoall_token_override


def mpi_barrier_token_override(in_args, new_token, comm):
    return (mpi_barrier_p.bind(new_token, comm=comm),)


token_override_registry[mpi_barrier_p] = mpi_barrier_token_override


def mpi_bcast_token_override(in_args, new_token, root, comm):
    x, _ = in_args
    return mpi_bcast_p.bind(x, new_token, root=root, comm=comm)


token_override_registry[mpi_bcast_p] = mpi_bcast_token_override


def mpi_gather_token_override(in_args, new_token, root, comm):
    x, _ = in_args
    return mpi_gather_p.bind(x, new_token, root=root, comm=comm)


token_override_registry[mpi_gather_p] = mpi_gather_token_override


def mpi_recv_token_override(in_args, new_token, source, tag, comm, status):
    x, _ = in_args
    return mpi_recv_p.bind(
        x, new_token, source=source, tag=tag, comm=comm, status=status
    )


token_override_registry[mpi_recv_p] = mpi_recv_token_override


def mpi_reduce_token_override(in_args, new_token, op, root, comm):
    x, _ = in_args
    return mpi_reduce_p.bind(x, new_token, op=op, root=root, comm=comm)


token_override_registry[mpi_reduce_p] = mpi_reduce_token_override


def mpi_scan_token_override(in_args, new_token, op, comm):
    x, _ = in_args
    return mpi_scan_p.bind(x, new_token, op=op, comm=comm)


token_override_registry[mpi_scan_p] = mpi_scan_token_override


def mpi_scatter_token_override(in_args, new_token, root, comm):
    x, _ = in_args
    return mpi_scatter_p.bind(x, new_token, root=root, comm=comm)


token_override_registry[mpi_scatter_p] = mpi_scatter_token_override


def mpi_send_token_override(in_args, new_token, dest, tag, comm):
    x, _ = in_args
    return (mpi_send_p.bind(x, new_token, dest=dest, tag=tag, comm=comm),)


token_override_registry[mpi_send_p] = mpi_send_token_override


def mpi_sendrecv_token_override(
    in_args, new_token, source, dest, sendtag, recvtag, comm, status, _must_transpose
):
    sendbuff, recvbuff, _ = in_args
    return mpi_sendrecv_p.bind(
        sendbuff,
        recvbuff,
        new_token,
        source=source,
        dest=dest,
        sendtag=sendtag,
        recvtag=recvtag,
        comm=comm,
        status=status,
        _must_transpose=_must_transpose,
    )


token_override_registry[mpi_sendrecv_p] = mpi_sendrecv_token_override

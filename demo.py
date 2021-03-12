import math
from collections import namedtuple
from functools import partial

import numpy as np
import tqdm

import jax
import jax.numpy as jnp

from mpi4py import MPI
import mpi4jax

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

npy = min(size, 2)
npx = size // npy
assert size == npx * npy
proc_idx = jnp.unravel_index(rank, (npy, npx))

# grid setup
nx = 400 // npx + 2
dx = 5e3
l_x = npx * nx * dx

ny = 200 // npy + 2
dy = 5e3
l_y = npy * ny * dy

x, y = (
    ((nx - 2) * proc_idx[1] + np.arange(-1, nx - 1)) * dx,
    ((ny - 2) * proc_idx[0] + np.arange(-1, ny - 1)) * dy,
)
yy, xx = np.meshgrid(y, x, indexing="ij")

# physical parameters
GRAVITY = 9.81
DEPTH = 100.0
CORIOLIS_F = 2e-4
CORIOLIS_BETA = 2e-11
CORIOLIS_PARAM = CORIOLIS_F + yy * CORIOLIS_BETA
LATERAL_VISCOSITY = 1e-3 * CORIOLIS_F * dx ** 2

# other parameters
periodic_boundary_x = True
linear_momentum_equation = False

ADAMS_BASHFORTH_A = 1.5 + 0.1
ADAMS_BASHFORTH_B = -(0.5 + 0.1)

dt = 0.125 * min(dx, dy) / np.sqrt(GRAVITY * DEPTH)

# plot parameters
plot_range = 10
plot_every = 100
max_quivers = 41

# initial conditions
u0 = 10 * jnp.exp(-((yy - 0.5 * l_y) ** 2) / (0.02 * l_x) ** 2)
v0 = jnp.zeros_like(u0)
h0 = (
    DEPTH
    # small perturbation
    + 0.2 * jnp.sin(xx / l_x * 10 * jnp.pi) * jnp.cos(yy / l_y * 8 * jnp.pi)
)


@partial(jax.jit, static_argnums=(1,))
def enforce_boundaries(arr, grid, token=None):
    assert grid in ("h", "u", "v")

    # start west, go clockwise
    send_order = (
        "west",
        "north",
        "east",
        "south",
    )

    # start east, go clockwise
    recv_order = (
        "east",
        "south",
        "west",
        "north",
    )

    overlap_slices_send = dict(
        south=(1, slice(None), Ellipsis),
        west=(slice(None), 1, Ellipsis),
        north=(-2, slice(None), Ellipsis),
        east=(slice(None), -2, Ellipsis),
    )

    overlap_slices_recv = dict(
        south=(0, slice(None), Ellipsis),
        west=(slice(None), 0, Ellipsis),
        north=(-1, slice(None), Ellipsis),
        east=(slice(None), -1, Ellipsis),
    )

    proc_neighbors = {
        "south": (proc_idx[0] - 1, proc_idx[1]) if proc_idx[0] > 0 else None,
        "west": (proc_idx[0], proc_idx[1] - 1) if proc_idx[1] > 0 else None,
        "north": (proc_idx[0] + 1, proc_idx[1]) if proc_idx[0] < npy - 1 else None,
        "east": (proc_idx[0], proc_idx[1] + 1) if proc_idx[1] < npx - 1 else None,
    }

    if periodic_boundary_x:
        if proc_idx[1] == 0:
            proc_neighbors["west"] = (proc_idx[0], npx - 1)

        if proc_idx[1] == npx - 1:
            proc_neighbors["east"] = (proc_idx[0], 0)

    if token is None:
        token = jax.lax.create_token()

    for send_dir, recv_dir in zip(send_order, recv_order):
        send_proc = proc_neighbors[send_dir]
        recv_proc = proc_neighbors[recv_dir]

        if send_proc is None and recv_proc is None:
            continue

        if send_proc is not None:
            send_proc = np.ravel_multi_index(send_proc, (npy, npx))

        if recv_proc is not None:
            recv_proc = np.ravel_multi_index(recv_proc, (npy, npx))

        recv_idx = overlap_slices_recv[recv_dir]
        recv_arr = jnp.empty_like(arr[recv_idx])

        send_idx = overlap_slices_send[send_dir]
        send_arr = arr[send_idx]

        if send_proc is None:
            recv_arr, token = mpi4jax.recv(
                recv_arr, source=recv_proc, comm=comm, token=token
            )
            arr = arr.at[recv_idx].set(recv_arr)
        elif recv_proc is None:
            token = mpi4jax.send(send_arr, dest=send_proc, comm=comm, token=token)
        else:
            recv_arr, token = mpi4jax.sendrecv(
                send_arr,
                recv_arr,
                source=recv_proc,
                dest=send_proc,
                comm=comm,
                token=token,
            )
            arr = arr.at[recv_idx].set(recv_arr)

    if not periodic_boundary_x and grid == "u" and proc_idx[1] == npx - 1:
        arr = arr.at[:, -2].set(0.0)

    if grid == "v" and proc_idx[0] == npy - 1:
        arr = arr.at[-2, :].set(0.0)

    return arr, token


ModelState = namedtuple("ModelState", "h, u, v, dh, du, dv")


@partial(jax.jit, static_argnums=(1,))
def shallow_water_step(state, first_step, token):
    h, u, v, dh, du, dv = state

    hc = jnp.pad(h[1:-1, 1:-1], 1, "edge")
    hc, token = enforce_boundaries(hc, "h", token)

    fe = jnp.empty_like(u)
    fn = jnp.empty_like(u)

    fe = fe.at[1:-1, 1:-1].set(0.5 * (hc[1:-1, 1:-1] + hc[1:-1, 2:]) * u[1:-1, 1:-1])
    fn = fn.at[1:-1, 1:-1].set(0.5 * (hc[1:-1, 1:-1] + hc[2:, 1:-1]) * v[1:-1, 1:-1])
    fe, token = enforce_boundaries(fe, "u", token)
    fn, token = enforce_boundaries(fn, "v", token)

    dh_new = dh.at[1:-1, 1:-1].set(
        -(fe[1:-1, 1:-1] - fe[1:-1, :-2]) / dx - (fn[1:-1, 1:-1] - fn[:-2, 1:-1]) / dy
    )

    if linear_momentum_equation:
        v_avg = 0.25 * (v[1:-1, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[:-2, 2:])
        du_new = du.at[1:-1, 1:-1].set(
            CORIOLIS_PARAM[1:-1, 1:-1] * v_avg
            - GRAVITY * (h[1:-1, 2:] - h[1:-1, 1:-1]) / dx
        )
        u_avg = 0.25 * (u[1:-1, 1:-1] + u[1:-1, :-2] + u[2:, 1:-1] + u[2:, :-2])
        dv_new = dv.at[1:-1, 1:-1].set(
            -CORIOLIS_PARAM[1:-1, 1:-1] * u_avg
            - GRAVITY * (h[2:, 1:-1] - h[1:-1, 1:-1]) / dy
        )

    else:  # nonlinear momentum equation
        q = jnp.empty_like(u)
        ke = jnp.empty_like(u)

        # planetary and relative vorticity
        q = q.at[1:-1, 1:-1].set(
            CORIOLIS_PARAM[1:-1, 1:-1]
            + ((v[1:-1, 2:] - v[1:-1, 1:-1]) / dx - (u[2:, 1:-1] - u[1:-1, 1:-1]) / dy)
        )
        # potential vorticity
        q = q.at[1:-1, 1:-1].mul(
            1.0 / (0.25 * (hc[1:-1, 1:-1] + hc[1:-1, 2:] + hc[2:, 1:-1] + hc[2:, 2:]))
        )
        q, token = enforce_boundaries(q, "h", token)

        du_new = du.at[1:-1, 1:-1].set(
            -GRAVITY * (h[1:-1, 2:] - h[1:-1, 1:-1]) / dx
            + 0.5
            * (
                q[1:-1, 1:-1] * 0.5 * (fn[1:-1, 1:-1] + fn[1:-1, 2:])
                + q[:-2, 1:-1] * 0.5 * (fn[:-2, 1:-1] + fn[:-2, 2:])
            )
        )
        dv_new = dv.at[1:-1, 1:-1].set(
            -GRAVITY * (h[2:, 1:-1] - h[1:-1, 1:-1]) / dy
            - 0.5
            * (
                q[1:-1, 1:-1] * 0.5 * (fe[1:-1, 1:-1] + fe[2:, 1:-1])
                + q[1:-1, :-2] * 0.5 * (fe[1:-1, :-2] + fe[2:, :-2])
            )
        )
        ke = ke.at[1:-1, 1:-1].set(
            0.5
            * (
                0.5 * (u[1:-1, 1:-1] ** 2 + u[1:-1, :-2] ** 2)
                + 0.5 * (v[1:-1, 1:-1] ** 2 + v[:-2, 1:-1] ** 2)
            )
        )
        ke, token = enforce_boundaries(ke, "h", token)

        du_new = du_new.at[1:-1, 1:-1].add(-(ke[1:-1, 2:] - ke[1:-1, 1:-1]) / dx)
        dv_new = dv_new.at[1:-1, 1:-1].add(-(ke[2:, 1:-1] - ke[1:-1, 1:-1]) / dy)

    if first_step:
        u = u.at[1:-1, 1:-1].add(dt * du_new[1:-1, 1:-1])
        v = v.at[1:-1, 1:-1].add(dt * dv_new[1:-1, 1:-1])
        h = h.at[1:-1, 1:-1].add(dt * dh_new[1:-1, 1:-1])
    else:
        u = u.at[1:-1, 1:-1].add(
            dt
            * (
                ADAMS_BASHFORTH_A * du_new[1:-1, 1:-1]
                + ADAMS_BASHFORTH_B * du[1:-1, 1:-1]
            )
        )
        v = v.at[1:-1, 1:-1].add(
            dt
            * (
                ADAMS_BASHFORTH_A * dv_new[1:-1, 1:-1]
                + ADAMS_BASHFORTH_B * dv[1:-1, 1:-1]
            )
        )
        h = h.at[1:-1, 1:-1].add(
            dt
            * (
                ADAMS_BASHFORTH_A * dh_new[1:-1, 1:-1]
                + ADAMS_BASHFORTH_B * dh[1:-1, 1:-1]
            )
        )

    h, token = enforce_boundaries(h, "h", token)
    u, token = enforce_boundaries(u, "u", token)
    v, token = enforce_boundaries(v, "v", token)

    if LATERAL_VISCOSITY > 0:
        # lateral friction
        fe = fe.at[1:-1, 1:-1].set(
            LATERAL_VISCOSITY * (u[1:-1, 2:] - u[1:-1, 1:-1]) / dx
        )
        fn = fn.at[1:-1, 1:-1].set(
            LATERAL_VISCOSITY * (u[2:, 1:-1] - u[1:-1, 1:-1]) / dy
        )
        fe, token = enforce_boundaries(fe, "u", token)
        fn, token = enforce_boundaries(fn, "v", token)

        u = u.at[1:-1, 1:-1].add(
            dt
            * (
                (fe[1:-1, 1:-1] - fe[1:-1, :-2]) / dx
                + (fn[1:-1, 1:-1] - fn[:-2, 1:-1]) / dy
            )
        )

        fe = fe.at[1:-1, 1:-1].set(
            LATERAL_VISCOSITY * (v[1:-1, 2:] - u[1:-1, 1:-1]) / dx
        )
        fn = fn.at[1:-1, 1:-1].set(
            LATERAL_VISCOSITY * (v[2:, 1:-1] - u[1:-1, 1:-1]) / dy
        )
        fe, token = enforce_boundaries(fe, "u", token)
        fn, token = enforce_boundaries(fn, "v", token)

        v = v.at[1:-1, 1:-1].add(
            dt
            * (
                (fe[1:-1, 1:-1] - fe[1:-1, :-2]) / dx
                + (fn[1:-1, 1:-1] - fn[:-2, 1:-1]) / dy
            )
        )

    return ModelState(h, u, v, dh_new, du_new, dv_new), token


@partial(jax.jit, static_argnums=(1,))
def do_multistep(state, num_steps):
    token = jax.lax.create_token()

    def loop_func(i, args):
        state, token = args
        return shallow_water_step(state, False, token)

    final_state, _ = jax.lax.fori_loop(0, num_steps, loop_func, (state, token))
    return final_state


def solve_shallow_water(t1, num_multisteps=10):
    # allocate arrays
    du, dv, dh = jnp.zeros((ny, nx)), jnp.zeros((ny, nx)), jnp.zeros((ny, nx))
    token = jax.lax.create_token()

    # initial conditions
    h, token = enforce_boundaries(h0, "h", token)
    u, token = enforce_boundaries(u0, "u", token)
    v, token = enforce_boundaries(v0, "v", token)

    state = ModelState(h, u, v, dh, du, dv)
    sol = [state]

    state, token = shallow_water_step(state, True, token)
    sol.append(state)
    t = dt

    pbar = tqdm.tqdm(
        total=math.ceil(t1 / dt),
        disable=rank != 0,
        unit="model day",
        initial=t / 86400,
        unit_scale=dt / 86400,
        bar_format=(
            "{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, "
            "{rate_fmt}{postfix}]"
        ),
    )

    # pre-compile JAX kernel
    do_multistep(state, num_multisteps)

    with pbar:
        while t < t1:
            state = do_multistep(state, num_multisteps)
            sol.append(state)

            t += dt * num_multisteps

            if t < t1:
                pbar.update(num_multisteps)

    return sol


def animate_shallow_water(xx, yy, sol):
    import matplotlib.pyplot as plt
    from matplotlib import animation

    x = xx[0, :]
    y = yy[:, 0]

    quiver_stride = (
        slice(1, -1, y.shape[0] // max_quivers),
        slice(1, -1, x.shape[0] // max_quivers),
    )

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    cs = ax.pcolormesh(
        0.5 * (x[:-1] + x[1:]) / 1e3,
        0.5 * (y[:-1] + y[1:]) / 1e3,
        sol[0].h[1:-1, 1:-1] - DEPTH,
        vmin=-plot_range,
        vmax=plot_range,
        cmap="RdBu_r",
    )
    cq = ax.quiver(
        xx[quiver_stride] / 1e3,
        yy[quiver_stride] / 1e3,
        sol[0].u[quiver_stride],
        sol[0].v[quiver_stride],
        clip_on=True,
    )
    t = ax.text(
        s="",
        x=0.05,
        y=0.95,
        ha="left",
        va="top",
        backgroundcolor=(1, 1, 1, 0.8),
        transform=ax.transAxes,
    )

    ax.set(
        aspect="equal",
        xlim=(xx.min() / 1e3, xx.max() / 1e3),
        ylim=(yy.min() / 1e3, yy.max() / 1e3),
        xlabel="$x$ (km)",
        ylabel="$y$ (km)",
    )

    plt.colorbar(cs, orientation="horizontal", label="$\\eta$ (m)")

    def animate(i):
        state = sol[i]
        time = plot_every * dt * i
        eta = state.h - DEPTH

        cs.set_array(eta[1:-1, 1:-1].flatten())
        cq.set_UVC(state.u[quiver_stride], state.v[quiver_stride])
        t.set_text(f"t = {time / 86400:.2f} days")
        return (cs, cq, t)

    anim = animation.FuncAnimation(  # noqa: F841
        fig, animate, frames=len(sol), interval=20, blit=True, repeat_delay=3_000
    )
    plt.show()


@jax.jit
def reassemble_array(arr):
    out = jnp.empty(((ny - 2) * npy, (nx - 2) * npx), dtype=arr.dtype)
    for i in range(size):
        proc_idx_i = np.unravel_index(i, (npy, npx))
        y_offset = proc_idx_i[0] * (ny - 2)
        x_offset = proc_idx_i[1] * (nx - 2)
        out_idx = (
            slice(y_offset, y_offset + ny - 2),
            slice(x_offset, x_offset + nx - 2),
        )
        out = out.at[out_idx].set(arr[i, 1:-1, 1:-1])

    return out


if __name__ == "__main__":
    sol = solve_shallow_water(10 * 86_400, plot_every)

    # full_sol_arr has shape (nproc, time, nvars, ny, nx)
    full_sol_arr, _ = mpi4jax.gather(jnp.asarray(sol), root=0)

    if rank == 0:
        x, y = (np.arange((nx - 2) * npx) * dx, np.arange((ny - 2) * npy) * dy)
        yy, xx = np.meshgrid(y, x, indexing="ij")

        full_sol_arr = jnp.moveaxis(full_sol_arr, 0, 2)
        full_sol = [
            ModelState(*(reassemble_array(xi) for xi in x)) for x in full_sol_arr
        ]
        animate_shallow_water(xx, yy, full_sol)

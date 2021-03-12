import os

from setuptools import setup, find_packages
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    HAS_CYTHON = False
else:
    HAS_CYTHON = True

CYTHON_SUBMODULE_NAME = "mpi4jax._src.xla_bridge"
CYTHON_SUBMODULE_PATH = "mpi4jax/_src/xla_bridge"


#######
# Utils
def search_on_path(filenames):
    for p in get_path("PATH"):
        for filename in filenames:
            full = os.path.join(p, filename)
            if os.path.exists(full):
                return os.path.abspath(full)
    return None


def print_warning(*lines):
    print("**************************************************")
    for line in lines:
        print("*** WARNING: %s" % line)
    print("**************************************************")


def get_path(key):
    return os.environ.get(key, "").split(os.pathsep)


# /end Utils
############


def mpi_info(cmd):
    import mpi4py

    config = mpi4py.get_config()
    cmd_compile = " ".join([config["mpicc"], "-show"])
    out_stream = os.popen(cmd_compile)
    flags = out_stream.read().strip()
    flags = flags.replace(",", " ").split()

    if cmd == "compile":
        startwith = "-I"
    elif cmd == "libdirs":
        startwith = "-L"
    elif cmd == "libs":
        startwith = "-l"

    out = []
    for flag in flags:
        if flag.startswith(startwith):
            out.append(flag[2:])
    return out


################
# Cuda detection

# Taken from CUPY (MIT License)
def get_cuda_path():
    nvcc_path = search_on_path(("nvcc", "nvcc.exe"))
    cuda_path_default = None
    if nvcc_path is not None:
        cuda_path_default = os.path.normpath(
            os.path.join(os.path.dirname(nvcc_path), "..")
        )

    cuda_path = os.environ.get("CUDA_PATH", "")  # Nvidia default on Windows
    if len(cuda_path) == 0:
        cuda_path = os.environ.get("CUDA_ROOT", "")  # Nvidia default on Windows

    if len(cuda_path) > 0 and cuda_path != cuda_path_default:
        print_warning(
            "nvcc path != CUDA_PATH",
            "nvcc path: %s" % cuda_path_default,
            "CUDA_PATH: %s" % cuda_path,
        )

    if os.path.exists(cuda_path):
        _cuda_path = cuda_path
    elif cuda_path_default is not None:
        _cuda_path = cuda_path_default
    elif os.path.exists("/usr/local/cuda"):
        _cuda_path = "/usr/local/cuda"
    else:
        _cuda_path = None

    return _cuda_path


def cuda_info(cmd):
    cuda_path = get_cuda_path()
    if not cuda_path:
        return []

    if cmd == "compile":
        incdir = os.path.join(cuda_path, "include")
        if os.path.isdir(incdir):
            return [incdir]

    if cmd == "libdirs":
        libdir = os.path.join(cuda_path, "lib64")
        if os.path.isdir(libdir):
            return [libdir]
        else:
            libdir = os.path.join(cuda_path, "lib")
            if os.path.isdir(libdir):
                return [libdir]

    if cmd == "libs":
        return ["cudart"]

    return []


# /end Cuda detection
#####################


def _env_to_bool(envvar):
    return os.environ.get(envvar, "").lower() in ("true", "1", "on")


def get_extensions():
    if HAS_CYTHON:
        ext_suffix = "pyx"
    else:
        ext_suffix = "c"

    activate_tracing = _env_to_bool("MPI4JAX_ENABLE_TRACING")

    if activate_tracing:
        macros = [("CYTHON_TRACE_NOGIL", "1")]
    else:
        macros = None

    extensions = [
        Extension(
            name=f"{CYTHON_SUBMODULE_NAME}.{mod}",
            sources=[f"{CYTHON_SUBMODULE_PATH}/{mod}.{ext_suffix}"],
            include_dirs=mpi_info("compile"),
            library_dirs=mpi_info("libdirs"),
            libraries=mpi_info("libs"),
            define_macros=macros,
        )
        for mod in ("mpi_xla_bridge", "mpi_xla_bridge_cpu")
    ]

    if cuda_info("compile"):
        extensions.append(
            Extension(
                name=f"{CYTHON_SUBMODULE_NAME}.mpi_xla_bridge_gpu",
                sources=[f"{CYTHON_SUBMODULE_PATH}/mpi_xla_bridge_gpu.{ext_suffix}"],
                include_dirs=mpi_info("compile") + cuda_info("compile"),
                library_dirs=mpi_info("libdirs") + cuda_info("libdirs"),
                libraries=mpi_info("libs") + cuda_info("libs"),
                define_macros=macros,
            )
        )

    if HAS_CYTHON:
        compiler_directives = {"linetrace": activate_tracing}
        extensions = cythonize(
            extensions,
            compiler_directives=compiler_directives,
            language_level=3,
        )

    return extensions


here = os.path.abspath(os.path.dirname(__file__))

# get the long description from the README file
with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mpi4jax",
    author="Filippo Vicentini",
    author_email="filippovicentini@gmail.com",
    long_description=long_description,
    url="https://github.com/PhilipVinc/mpi4jax",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    ext_modules=get_extensions(),
    use_scm_version=dict(
        write_to="mpi4jax/_version.py",
    ),
    python_requires=">=3.6",
    install_requires=["jax", "jaxlib>=0.1.62", "mpi4py>=3.0.1", "numpy"],
    extras_require={"dev": ["pytest", "black", "flake8==3.8.3", "pre-commit>=2"]},
)

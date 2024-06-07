import os
import sys
import shlex

import importlib.util
import pathlib
import fnmatch

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext

here = os.path.abspath(os.path.dirname(__file__))

# ensure vendored versioneer is on path
sys.path.insert(0, here)
import versioneer  # noqa: E402

try:
    from Cython.Build import cythonize
except ImportError:
    HAS_CYTHON = False
else:
    HAS_CYTHON = True

try:
    import mpi4py
except ImportError:
    HAS_MPI4PY = False
else:
    HAS_MPI4PY = True


##############
# Requirements

JAX_MINIMUM_VERSION = "0.4.5"

BASE_DEPENDENCIES = ["mpi4py>=3.0.1", "numpy", f"jax>={JAX_MINIMUM_VERSION}"]

DEV_DEPENDENCIES = [
    "pytest>=6",
    "pytest-cov>=2.10.1",
    "coverage[toml]>=5",
    "pre-commit",
    "black==23.9; python_version >= '3.8.0'",
    "flake8==3.9.2",
    "tqdm>=4.52",
]

CYTHON_SUBMODULE_NAME = "mpi4jax._src.xla_bridge"
CYTHON_SUBMODULE_PATH = "mpi4jax/_src/xla_bridge"


#######
# Utils
def search_on_path(filenames):
    path_dirs = os.getenv("PATH", "").split(os.pathsep)

    for p in path_dirs:
        for filename in filenames:
            full_path = os.path.join(p, filename)
            if os.path.exists(full_path):
                return os.path.abspath(full_path)

    return None


def print_warning(*lines):
    print("**************************************************")
    for line in lines:
        print("*** WARNING: %s" % line)
    print("**************************************************")


# /end Utils
############


class custom_build_ext(build_ext):
    def build_extensions(self):
        config = mpi4py.get_config()
        mpi_compiler = os.environ.get("MPI4JAX_BUILD_MPICC", config["mpicc"])
        mpi_cmd = shlex.split(mpi_compiler)

        for exe in ("compiler", "compiler_so", "compiler_cxx", "linker_so"):
            # peel off compiler executable but keep flags
            current_flags = getattr(self.compiler, exe)[1:]
            self.compiler.set_executable(exe, [*mpi_cmd, *current_flags])

        build_ext.build_extensions(self)


################
# Cuda detection


# partly taken from JAX
# https://github.com/google/jax/blob/4cca2335220dcc953edd2ac764b2387e53527495/jax/_src/lib/__init__.py#L129
def get_cuda_paths_from_nvidia_pypi():
    # try to check if nvidia-cuda-nvcc-cu* is installed
    # we need to get the site-packages of this install. to do so we use
    # mpi4py which must be installed
    mpi4py_spec = importlib.util.find_spec("mpi4py")
    depot_path = pathlib.Path(os.path.dirname(mpi4py_spec.origin)).parent

    # If the pip package nvidia-cuda-nvcc-cu11 is installed, it should have
    # both of the things XLA looks for in the cuda path, namely bin/ptxas and
    # nvvm/libdevice/libdevice.10.bc
    #
    # The files are split in two sets of directories, so we return both
    maybe_cuda_paths = [
        depot_path / "nvidia" / "cuda_nvcc",
        depot_path / "nvidia" / "cuda_runtime",
    ]
    if all(p.is_dir() for p in maybe_cuda_paths):
        return [str(p) for p in maybe_cuda_paths]
    else:
        return []


# Taken from CUPY (MIT License)
def get_cuda_path():
    nvcc_path = search_on_path(("nvcc", "nvcc.exe"))
    cuda_path_default = None
    if nvcc_path is not None:
        cuda_path_default = os.path.normpath(
            os.path.join(os.path.dirname(nvcc_path), "..")
        )

    cuda_path = os.getenv("CUDA_PATH", "")  # Nvidia default on Windows
    if len(cuda_path) == 0:
        cuda_path = os.getenv("CUDA_ROOT", "")  # Nvidia default on Windows

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


def get_sycl_path():
    sycl_path = os.getenv("CMPLR_ROOT", "")
    if len(sycl_path) > 0 and os.path.exists(sycl_path):
        _sycl_path = sycl_path
    elif os.path.exists("/opt/intel/oneapi/compiler/latest/"):
        _sycl_path = "/opt/intel/oneapi/compiler/latest/"
    else:
        _sycl_path = None

    return _sycl_path


def get_sycl_info():
    sycl_info = {"compile": [], "libdirs": [], "libs": []}
    sycl_path = get_sycl_path()
    if not sycl_path:
        return sycl_info

    include_suffixes = [
        "linux/include/",
        "linux/include/sycl",
        "include/",
        "include/sycl",
    ]

    for inc_suffix in include_suffixes:
        incdir = os.path.join(sycl_path, inc_suffix)
        if os.path.isdir(incdir):
            sycl_info["compile"].append(incdir)

    libdir_suffixes = [
        "linux/lib/",
        "lib/",
    ]
    for libdir_suffix in libdir_suffixes:
        lib_dir = os.path.join(sycl_path, libdir_suffix)
        if os.path.isdir(lib_dir):
            sycl_info["libdirs"].append(lib_dir)

    sycl_info["libs"].append("sycl")
    return sycl_info


sycl_info = get_sycl_info()


def find_files(bases, pattern):
    """Return list of files matching pattern in base folders and subfolders."""
    if isinstance(bases, (str, pathlib.Path)):
        bases = [bases]

    result = []
    for base in bases:
        for root, dirs, files in os.walk(base):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
    return result


def get_cuda_info():
    cuda_info = {"compile": [], "libdirs": [], "libs": [], "rpaths": []}

    # First check if the nvidia-cuda-nvcc-cu* package is installed. We ignore CUDA_ROOT
    # because that is the same behaviour of jax.
    cuda_paths = get_cuda_paths_from_nvidia_pypi()

    # If not, try to find the CUDA_PATH by hand
    if len(cuda_paths) > 0:
        nvidia_pypi_package = True
    else:
        nvidia_pypi_package = False
        _cuda_path = get_cuda_path()
        if _cuda_path is None:
            cuda_paths = []
        else:
            cuda_paths = [_cuda_path]

    if len(cuda_paths) == 0:
        return cuda_info

    for cuda_path in cuda_paths:
        incdir = os.path.join(cuda_path, "include")
        if os.path.isdir(incdir):
            cuda_info["compile"].append(incdir)

        for libdir in ("lib64", "lib"):
            full_dir = os.path.join(cuda_path, libdir)
            if os.path.isdir(full_dir):
                cuda_info["libdirs"].append(full_dir)

    # We need to link against libcudart.so
    #   - If we are using standard CUDA installations, we simply add a link flag to
    #     libcudart.so
    #   - If we are using the nvidia-cuda-nvcc-cu* package, we need to find the exact
    #     version of libcudart.so to link against because the the package does not provide
    #     a generic binding to libcudart.so but only libcudart.so.XX.
    #
    # Moreover, if we are using nvidia-cuda-nvcc we must add @rpath (runtime search paths)
    # because we do not expect the user to set LD_LIBRARY_PATH to the nvidia-cuda-nvcc
    # package.
    if not nvidia_pypi_package:
        cuda_info["libs"].append("cudart")
    else:
        possible_libcudart = find_files(cuda_paths, "libcudart.so*")

        if "libcudart.so" in possible_libcudart:
            # If generic symlink is present, use standard linker flag.
            # In theory with nvidia-cuda-nvcc-cu12 we should never reach this point
            # But in the future they might fix it.
            cuda_info["libs"].append("cudart")
        elif len(possible_libcudart) > 0:
            # This should be the standard case for nvidia-cuda-nvcc-cu*
            # where we find a library libcudart.so.XX . The syntax to link to a
            # specific version is -l:libcudart.so.XX
            # We arbitrarily choose the first one
            # and we add the runtime search path accordingly
            lib_to_link = possible_libcudart[0]
            cuda_info["libs"].append(f":{os.path.basename(lib_to_link)}")
            cuda_info["rpaths"].append(os.path.dirname(lib_to_link))
        else:
            # If we cannot find libcudart.so, we cannot build the extension
            # This should never happen with nvidia-cuda-nvcc-cu* package
            cuda_info["libs"].append("cudart")

    print("\n\nCUDA INFO:", cuda_info, "\n\n")
    return cuda_info


cuda_info = get_cuda_info()

# /end Cuda detection
#####################


def get_extensions():
    cmd = sys.argv[1]
    require_extensions = any(
        cmd.startswith(subcmd) for subcmd in ("install", "build", "bdist", "develop")
    )

    if not HAS_MPI4PY or not HAS_CYTHON:
        # this should only happen when using python setup.py
        # or pip install --no-build-isolation
        if require_extensions:
            raise RuntimeError("Building mpi4jax requires Cython and mpi4py")
        else:
            return []

    extensions = [
        Extension(
            name=f"{CYTHON_SUBMODULE_NAME}.{mod}",
            sources=[f"{CYTHON_SUBMODULE_PATH}/{mod}.pyx"],
        )
        for mod in ("mpi_xla_bridge", "mpi_xla_bridge_cpu", "device_descriptors")
    ]

    if sycl_info["compile"] and sycl_info["libdirs"]:
        extensions.append(
            Extension(
                name=f"{CYTHON_SUBMODULE_NAME}.mpi_xla_bridge_xpu",
                sources=[f"{CYTHON_SUBMODULE_PATH}/mpi_xla_bridge_xpu.pyx"],
                include_dirs=sycl_info["compile"],
                library_dirs=sycl_info["libdirs"],
                libraries=sycl_info["libs"],
                language="c++",
                # This macro instructs C++ compiler to ignore potential existence of
                # OpenMPI C++ bindings which are deprecated
                define_macros=[("OMPI_SKIP_MPICXX", "1")],
            )
        )
    else:
        print_warning(
            "SYCL (Intel Basekit) path not found",
            "(XPU extensions will not be built)",
        )

    if cuda_info["compile"] and cuda_info["libdirs"]:
        extra_extension_args = {}
        if len(cuda_info["rpaths"]) > 0:
            extra_extension_args["runtime_library_dirs"] = cuda_info["rpaths"]

        extensions.append(
            Extension(
                name=f"{CYTHON_SUBMODULE_NAME}.mpi_xla_bridge_gpu",
                sources=[f"{CYTHON_SUBMODULE_PATH}/mpi_xla_bridge_gpu.pyx"],
                include_dirs=cuda_info["compile"],
                library_dirs=cuda_info["libdirs"],
                libraries=cuda_info["libs"],
                **extra_extension_args,
            )
        )
    else:
        print_warning("CUDA path not found", "(GPU extensions will not be built)")

    if HAS_CYTHON:
        extensions = cythonize(
            extensions,
            language_level=3,
        )

    return extensions


# get the long description from the README file
with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()


cmdclass = versioneer.get_cmdclass()
cmdclass.update(build_ext=custom_build_ext)

setup(
    name="mpi4jax",
    author="Filippo Vicentini",
    author_email="filippovicentini@gmail.com",
    description=(
        "Zero-copy MPI communication of JAX arrays, "
        "for turbo-charged HPC applications in Python ⚡"
    ),
    long_description=long_description,
    url="https://github.com/mpi4jax/mpi4jax",
    license="MIT",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Development Status :: 3 - Alpha",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(),
    ext_modules=get_extensions(),
    python_requires=">=3.8",
    install_requires=BASE_DEPENDENCIES,
    extras_require={
        "dev": DEV_DEPENDENCIES,
    },
    package_data={"mpi4jax": ["_src/_latest_jax_version.txt"]},
    zip_safe=False,
)

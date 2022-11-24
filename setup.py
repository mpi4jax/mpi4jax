import os
import sys
import shlex

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

JAX_MINIMUM_VERSION = "0.3.25"

BASE_DEPENDENCIES = ["mpi4py>=3.0.1", "numpy", f"jax>={JAX_MINIMUM_VERSION}"]

DEV_DEPENDENCIES = [
    "pytest>=6",
    "pytest-cov>=2.10.1",
    "coverage[toml]>=5",
    "pre-commit",
    "black==21.6b0",
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


def get_cuda_info():
    cuda_info = {"compile": [], "libdirs": [], "libs": []}
    cuda_path = get_cuda_path()
    if not cuda_path:
        return cuda_info

    incdir = os.path.join(cuda_path, "include")
    if os.path.isdir(incdir):
        cuda_info["compile"].append(incdir)

    for libdir in ("lib64", "lib"):
        full_dir = os.path.join(cuda_path, libdir)
        if os.path.isdir(full_dir):
            cuda_info["libdirs"].append(full_dir)

    cuda_info["libs"].append("cudart")
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
        for mod in ("mpi_xla_bridge", "mpi_xla_bridge_cpu")
    ]

    if cuda_info["compile"] and cuda_info["libdirs"]:
        extensions.append(
            Extension(
                name=f"{CYTHON_SUBMODULE_NAME}.mpi_xla_bridge_gpu",
                sources=[f"{CYTHON_SUBMODULE_PATH}/mpi_xla_bridge_gpu.pyx"],
                include_dirs=cuda_info["compile"],
                library_dirs=cuda_info["libdirs"],
                libraries=cuda_info["libs"],
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
    python_requires=">=3.7",
    install_requires=BASE_DEPENDENCIES,
    extras_require={
        "dev": DEV_DEPENDENCIES,
    },
    package_data={"mpi4jax": ["_src/_latest_jax_version.txt"]},
    zip_safe=False,
)

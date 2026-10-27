# Copyright © 2026 Apple Inc.

import os
from importlib.metadata import version
from pathlib import Path

from mlx import extension
from setuptools import setup

if version("mlx") != "0.32.4":
    raise RuntimeError("Build spark-mlx-ext with official mlx==0.32.4")

source = Path(__file__).resolve().parent
repository = next(
    (path for path in (source, *source.parents) if (path / ".git").exists()),
    source,
)
build_root = Path(
    os.environ.get("SPARK_MLX_BUILD_ROOT", repository.parent / ".build/spark-mlx-ext")
).resolve()
metadata_root = build_root / "metadata"
metadata_root.mkdir(parents=True, exist_ok=True)

setup(
    ext_modules=[extension.CMakeExtension("spark_mlx_ext._ext")],
    cmdclass={"build_ext": extension.CMakeBuild},
    options={
        "build": {"build_base": str(build_root / "build")},
        "egg_info": {"egg_base": str(metadata_root)},
        "bdist_wheel": {
            "bdist_dir": str(build_root / "wheel-staging"),
            "dist_dir": str(build_root / "wheels"),
        },
        "sdist": {"dist_dir": str(build_root / "wheels")},
    },
)

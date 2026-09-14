"""Small helpers used while configuring the CUDA extension build."""

from __future__ import annotations

import os
import site
import sysconfig
from pathlib import Path
from typing import Iterable


def _site_package_dirs(explicit: Iterable[str | os.PathLike[str]] | None) -> list[Path]:
    if explicit is not None:
        candidates = [Path(path) for path in explicit]
    else:
        candidates = []
        try:
            candidates.extend(Path(path) for path in site.getsitepackages())
        except AttributeError:
            pass
        candidates.append(Path(sysconfig.get_paths()["purelib"]))

    result: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def get_cuda_include_dirs(
    cuda_home: str | os.PathLike[str] | None = None,
    site_packages: Iterable[str | os.PathLike[str]] | None = None,
) -> list[Path]:
    """Return toolkit and pip-installed CUDA component include directories.

    PyTorch CUDA wheels install headers such as ``cusparse.h`` below
    ``site-packages/nvidia/<component>/include``.  ``CUDAExtension`` adds the
    toolkit include directory, but does not discover those wheel directories.
    Keeping both locations makes the extension build work with either a full
    toolkit or a pip-provided CUDA runtime/development stack.
    """

    root = Path(cuda_home or os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
    candidates = [
        root / "include",
        root / "targets" / "x86_64-linux" / "include",
    ]

    for package_dir in _site_package_dirs(site_packages):
        nvidia_dir = package_dir / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        candidates.extend(
            include_dir
            for component_dir in sorted(nvidia_dir.iterdir())
            if component_dir.is_dir()
            for include_dir in [component_dir / "include"]
        )

    result: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_dir() and resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result

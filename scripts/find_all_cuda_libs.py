#!/usr/bin/env python3
"""Comprehensive search for CUDA library directories in site-packages."""

from pathlib import Path
import sys
import site

def find_cuda_lib_directories():
    """Search site-packages for CUDA library directories."""
    lib_dirs = set()

    # Get all site-packages directories
    site_packages = site.getsitepackages()
    if hasattr(site, 'getusersitepackages'):
        site_packages.append(site.getusersitepackages())

    print(f"Searching site-packages directories:", file=sys.stderr)
    for sp in site_packages:
        print(f"  - {sp}", file=sys.stderr)

    # CUDA library prefixes to search for
    cuda_prefixes = (
        "libcudnn",
        "libcublas",
        "libcublasLt",
        "libcuda",
        "libnvrtc",
        "libcurand",
        "libcusolver",
        "libcusparse",
        "libcufft",
    )

    for sp_dir in site_packages:
        sp_path = Path(sp_dir)
        if not sp_path.exists():
            continue

        # Search for .so files in nvidia subdirectories
        # nvidia packages often install to nvidia/cudnn/lib, nvidia/cublas/lib, etc
        nvidia_dirs = list(sp_path.glob("nvidia/*/lib"))
        for lib_dir in nvidia_dirs:
            if lib_dir.is_dir():
                # Check if it contains any .so files
                so_files = list(lib_dir.glob("*.so*"))
                if so_files:
                    print(f"Found nvidia lib dir: {lib_dir} ({len(so_files)} .so files)", file=sys.stderr)
                    lib_dirs.add(str(lib_dir.resolve()))

        # Also search torch/lib directory
        torch_lib = sp_path / "torch" / "lib"
        if torch_lib.exists():
            so_files = list(torch_lib.glob("*.so*"))
            if so_files:
                print(f"Found torch lib dir: {torch_lib} ({len(so_files)} .so files)", file=sys.stderr)
                lib_dirs.add(str(torch_lib.resolve()))

        # Search torchvision.libs if it exists
        torch_vision_libs = sp_path / "torchvision.libs"
        if torch_vision_libs.exists():
            so_files = list(torch_vision_libs.glob("*.so*"))
            if so_files:
                print(f"Found torchvision.libs: {torch_vision_libs} ({len(so_files)} .so files)", file=sys.stderr)
                lib_dirs.add(str(torch_vision_libs.resolve()))

        # Direct search for CUDA .so files anywhere in site-packages
        # This is a broader search for files like libcudnn.so.9
        for prefix in cuda_prefixes:
            matches = list(sp_path.glob(f"**/{prefix}.so*"))
            for match in matches:
                parent = match.parent.resolve()
                if parent not in lib_dirs:
                    print(f"Found {prefix} in: {parent}", file=sys.stderr)
                    lib_dirs.add(str(parent))

    # Also check vendor/cuda_stubs if it exists
    vendor_stubs = Path("vendor/cuda_stubs")
    if vendor_stubs.exists():
        print(f"Found vendor stubs: {vendor_stubs}", file=sys.stderr)
        lib_dirs.add(str(vendor_stubs.resolve()))

    return sorted(lib_dirs)

if __name__ == "__main__":
    dirs = find_cuda_lib_directories()
    if not dirs:
        print("ERROR: No CUDA library directories found!", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(dirs)} CUDA library directories:", file=sys.stderr)
    for d in dirs:
        print(d)

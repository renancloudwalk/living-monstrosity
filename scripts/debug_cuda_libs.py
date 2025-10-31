#!/usr/bin/env python3
"""Debug script to find nvidia-cudnn-cu12 package files and library locations."""

import importlib.metadata as md
from pathlib import Path

print("=" * 60)
print("Searching for nvidia-cudnn-cu12 package...")
print("=" * 60)

try:
    dist = md.distribution("nvidia-cudnn-cu12")
    print(f"✓ Found package: {dist.name} {dist.version}")
    print(f"  Location: {dist.locate_file('.')}")

    files = dist.files or []
    print(f"  Total files: {len(files)}")

    # Find .so files
    so_files = [f for f in files if ".so" in str(f)]
    print(f"  Files with .so: {len(so_files)}")

    if so_files:
        print("\nFirst 10 .so files:")
        for f in so_files[:10]:
            full_path = dist.locate_file(f)
            print(f"    {f}")
            print(f"      → {full_path}")
            print(f"      → parent: {Path(full_path).parent}")

        # Find the lib directory
        first_so = so_files[0]
        lib_dir = Path(dist.locate_file(first_so)).parent
        print(f"\n✓ Detected lib directory: {lib_dir}")
    else:
        print("\n✗ No .so files found in package!")
        print("\nShowing first 20 files:")
        for f in files[:20]:
            print(f"    {f}")

except md.PackageNotFoundError:
    print("✗ nvidia-cudnn-cu12 package NOT installed!")
    print("\nInstalled nvidia packages:")
    for dist in md.distributions():
        if "nvidia" in dist.name.lower():
            print(f"  - {dist.name} {dist.version}")

print("\n" + "=" * 60)
print("Checking all NVIDIA CUDA packages...")
print("=" * 60)

cuda_packages = [
    "nvidia-cudnn-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "torch",
]

for pkg_name in cuda_packages:
    try:
        dist = md.distribution(pkg_name)
        files = dist.files or []
        so_files = [f for f in files if ".so" in str(f)]

        if so_files:
            first_so = so_files[0]
            lib_dir = Path(dist.locate_file(first_so)).parent
            print(f"✓ {pkg_name:30s} → {lib_dir}")
        else:
            print(f"  {pkg_name:30s} → (no .so files)")
    except md.PackageNotFoundError:
        print(f"✗ {pkg_name:30s} → NOT INSTALLED")

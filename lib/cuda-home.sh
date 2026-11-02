#!/usr/bin/env bash
# A CUDA_HOME from NVIDIA's toolkit wheels (lib/cuda-wheels.txt: nvidia-cuda-nvcc, -crt, -runtime,
# nvidia-nvvm, -cccl, all 13.1), for machines without a system toolkit. The wheels put nvcc,
# the headers, nvvm and the libraries in one directory but in the pip layout (lib, no lib64, no
# unversioned .so names, no driver stub); the JIT builders (DeepGEMM, TileLang, FlashInfer, sgl-kernel,
# torch's cpp_extension) expect a toolkit's: bin/, include/, nvvm/, lib64/ with -lcudart linkable,
# lib64/stubs/libcuda.so. So this assembles one from symlinks.
# The wheels go into build/cuda-wheels (lib/cuda-wheels.txt), not the venv: torch pins the venv's
# nvidia-cuda-runtime to 13.0.96, and the compiler, its crt headers and cicc (nvidia-nvvm) must be of
# one minor version (cccl's header check; CUDA 13.0's crt clashes with glibc 2.43).
# usage (sourced): cuda_home_from_wheels <venv python> <build dir>; prints <build dir>/cuda-home, or
# nothing if the wheels could not be installed. cuda_home_assemble <wheels nvidia/cu13 dir> <out dir>
# does the symlink layout alone.
cuda_home_from_wheels() {
  local py="$1" build="$2" here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if ! [[ -x "$build/cuda-wheels/nvidia/cu13/bin/nvcc" && -e "$build/cuda-wheels/.from" \
          && "$(cat "$build/cuda-wheels/.from")" == "$(sha256sum "$here/lib/cuda-wheels.txt" | cut -c1-12)" ]]; then
    rm -rf "$build/cuda-wheels"; mkdir -p "$build/cuda-wheels"
    if command -v uv >/dev/null; then
      uv pip install --python "$py" --target "$build/cuda-wheels" -r "$here/lib/cuda-wheels.txt" >&2 || return 0
    else
      "$py" -m pip install --target "$build/cuda-wheels" -r "$here/lib/cuda-wheels.txt" >&2 || return 0
    fi
    sha256sum "$here/lib/cuda-wheels.txt" | cut -c1-12 > "$build/cuda-wheels/.from"
  fi
  cuda_home_assemble "$build/cuda-wheels/nvidia/cu13" "$build/cuda-home"
}

cuda_home_assemble() {
  local pip_cuda="$1" out="$2"
  [[ -x "$pip_cuda/bin/nvcc" ]] || return 0
  if [[ -e "$out/.from" && "$(cat "$out/.from")" == "$pip_cuda $(cat "$pip_cuda/bin/nvcc.profile" 2>/dev/null | md5sum | cut -c1-12)" ]]; then
    echo "$out"; return 0
  fi
  rm -rf "$out"; mkdir -p "$out/lib64/stubs"
  ln -s "$pip_cuda/bin" "$out/bin"
  ln -s "$pip_cuda/include" "$out/include"
  ln -s "$pip_cuda/nvvm" "$out/nvvm"
  local f base
  for f in "$pip_cuda"/lib/*; do
    base="$(basename "$f")"
    ln -s "$f" "$out/lib64/$base"
    # the unversioned name the linkers ask for (-lcudart): libcudart.so -> libcudart.so.13
    if [[ "$base" =~ ^(lib[A-Za-z0-9_]+\.so)\.[0-9]+$ ]]; then
      [[ -e "$out/lib64/${BASH_REMATCH[1]}" ]] || ln -s "$f" "$out/lib64/${BASH_REMATCH[1]}"
    fi
  done
  # the driver API for linking (-lcuda): the driver's own library
  local drv
  drv="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 / {print $NF; exit}')"
  [[ -n "$drv" ]] && ln -s "$drv" "$out/lib64/stubs/libcuda.so"
  echo "$pip_cuda $(cat "$pip_cuda/bin/nvcc.profile" 2>/dev/null | md5sum | cut -c1-12)" > "$out/.from"
  echo "$out"
}

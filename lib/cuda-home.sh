#!/usr/bin/env bash
# A CUDA_HOME from the venv's own toolkit wheels (requirements.txt: nvidia-cuda-nvcc, -crt, -runtime,
# nvidia-nvvm, -cccl under nvidia/cu13), for machines without a system toolkit. The wheels put nvcc,
# the headers, nvvm and the libraries in one directory but in the pip layout (lib, no lib64, no
# unversioned .so names, no driver stub); the JIT builders (DeepGEMM, TileLang, FlashInfer, sgl-kernel,
# torch's cpp_extension) expect a toolkit's: bin/, include/, nvvm/, lib64/ with -lcudart linkable,
# lib64/stubs/libcuda.so. So this assembles one from symlinks.
# usage (sourced): cuda_home_from_venv <venv> <out dir>; prints the out dir, or nothing if the venv
# has no nvcc.
cuda_home_from_venv() {
  local venv="$1" out="$2" pip_cuda
  pip_cuda="$(ls -d "$venv"/lib/python3.*/site-packages/nvidia/cu13 2>/dev/null | head -1)"
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

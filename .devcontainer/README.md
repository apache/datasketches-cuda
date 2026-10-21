# Datasketches CUDA Dev Container

The default container uses the GPU test toolchain. The CPU build configurations
are also available under `.devcontainer/cuda12.0-gcc11/` and
`.devcontainer/cuda13.1-gcc14/`. Their image and compiler selections come from
`ci/matrix.json`. After updating the matrix or shared devcontainer settings,
regenerate and check the files with:

```bash
python3 .devcontainer/generate.py
python3 .devcontainer/generate.py --check
```

## VS Code

Open the repository and use **Dev Containers: Reopen in Container**. GPU access
is optional for configuration and compilation, but runtime tests need a host
NVIDIA driver plus the NVIDIA Container Toolkit.

## Dev Container CLI

From a regular checkout:

```bash
devcontainer up \
  --workspace-folder "$PWD" \
  --config .devcontainer/devcontainer.json \
  --gpu-availability detect \
  --skip-post-attach

devcontainer exec \
  --workspace-folder "$PWD" \
  --config .devcontainer/devcontainer.json \
  bash -lc 'cmake -S . -B build/devcontainer -DCMAKE_CUDA_ARCHITECTURES=80 && cmake --build build/devcontainer -j"$(nproc)"'
```

When launching from a linked Git worktree, also bind-mount the Git common
directory so Git works inside the container:

```bash
git_common_dir="$(cd "$(git rev-parse --git-common-dir)" && pwd)"

devcontainer up \
  --workspace-folder "$PWD" \
  --config .devcontainer/devcontainer.json \
  --gpu-availability detect \
  --skip-post-attach \
  --mount type=bind,source="$git_common_dir",target="$git_common_dir"
```

## Build And Test

```bash
cmake -S . -B build/devcontainer -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build/devcontainer -j"$(nproc)"
ctest --test-dir build/devcontainer --output-on-failure
```

# Datasketches CUDA Dev Container

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

#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MATRIX = ROOT.parent / "ci" / "matrix.json"


def with_image(version, config):
    result = dict(config)
    result["image"] = (
        f"rapidsai/devcontainers:{version}-cpp-gcc{config['gcc']}"
        f"-cuda{config['cuda']}-{config['os']}"
    )
    return result


def render(template, config):
    result = copy.deepcopy(template)
    suffix = f"cuda{config['cuda']}-gcc{config['gcc']}"
    result["name"] = f"datasketches-cuda-{suffix}"
    result["image"] = config["image"]
    result["runArgs"][result["runArgs"].index("--name") + 1] = (
        f"${{localEnv:USER:anon}}-${{localWorkspaceFolderBasename}}-datasketches-{suffix}"
    )
    result["containerEnv"].update(
        {
            "CC": f"gcc-{config['gcc']}",
            "CXX": f"g++-{config['gcc']}",
            "CUDAHOSTCXX": f"g++-{config['gcc']}",
            "DEVCONTAINER_NAME": suffix,
            "CCCL_CUDA_VERSION": config["cuda"],
            "CCCL_HOST_COMPILER_VERSION": config["gcc"],
        }
    )
    return json.dumps(result, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify generated files are current")
    parser.add_argument("--github-output", type=Path, help="Write resolved CI matrix outputs")
    args = parser.parse_args()

    matrix = json.loads(MATRIX.read_text())
    builds = [with_image(matrix["rapids_version"], config) for config in matrix["build"]]
    test_build = next(config for config in builds if config["id"] == matrix["test"]["build"])
    test = {**test_build, "arch": matrix["test"]["arch"]}
    if test["arch"] not in test_build["arch"].split(";"):
        raise ValueError("GPU test architecture must be included in its build configuration")
    template = json.loads((ROOT / "devcontainer.json").read_text())
    expected = {ROOT / "devcontainer.json": render(template, test)}
    for config in builds:
        name = f"cuda{config['cuda']}-gcc{config['gcc']}"
        expected[ROOT / name / "devcontainer.json"] = render(template, config)

    stale = set(ROOT.glob("cuda*-gcc*/devcontainer.json")) - set(expected)
    for path in sorted(stale):
        print(f"Stale generated devcontainer: {path}", file=sys.stderr)

    outdated = bool(stale)
    for path, content in expected.items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                print(f"Outdated generated devcontainer: {path}", file=sys.stderr)
                outdated = True
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if args.github_output and not outdated:
        with args.github_output.open("a") as output:
            output.write(f"builds={json.dumps(builds, separators=(',', ':'))}\n")
            output.write(f"test_image={test['image']}\n")
    return int(outdated)


if __name__ == "__main__":
    sys.exit(main())

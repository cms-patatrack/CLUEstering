#!/bin/python3

import os
import subprocess
import sys

def parse_args() -> tuple:
    path = sys.argv[2]
    flags = []
    values = []
    for arg in sys.argv[2:]:
        parsed = arg.split('=')
        flags.append(parsed[0])
        values.append(parsed[1])

    return (flags, values)

if sys.argv[1] == "-h" or sys.argv[1] == "--help" or len(sys.argv) < 3:
    print("Usage: $0 <path> <version> <fork>")
    sys.exit(0)

flags, values = parse_args()
if '--fork' in flags:
    fork = values[flags.index('--fork')]
else:
    fork = 'cms-patatrack'
if '--commit' in flags:
    is_commit = True
    is_branch = False
    version = values[flags.index('--commit')]
elif '--branch' in flags:
    is_commit = False
    is_branch = True
    version = values[flags.index('--branch')]

clue_version = f"CLUEstering_{fork}_{version}"

if is_commit:
    print(f"Fetching CLUEstering commit {version} from {fork}")
if is_branch:
    print(f"Fetching CLUEstering branch {version} from {fork}")

repo_url = f"https://github.com/{fork}/CLUEstering.git"
# Checkout the repo
os.system("mkdir -p " + clue_version)

clone = subprocess.run(["git",
                        "clone",
                        repo_url,
                        f"{clue_version}/Debug",
                        "--recursive",
                        "--depth=1"], capture_output=True, text=True)
if clone.returncode != 0:
    print(f"Error: {clone.stderr}")
    print((f"Failed to clone the repo {repo_url}. Check that the insersted repository,"
           "commit or branch are correct"))
    sys.exit(1)
os.system("cp -r " +
          f"{clue_version}/Debug " +
          f"{clue_version}/Release")
print(f"Cloned {clue_version} successfully")
print('')
print("Compiling the Debug version of the library")
os.chdir(f"{clue_version}/Debug")
# Compile the debug version
compile_debug = subprocess.run(["cmake",
                                "-B",
                                "build",
                                "-DCMAKE_BUILD_TYPE=Debug"],
                                capture_output=True,
                                text=True)
if compile_debug.returncode != 0:
    print(f"Error: {compile_debug.stderr}")
    print("Failed to compile the debug version of the project")
    sys.exit(1)
compile_debug = subprocess.run(["cmake",
                                "--build",
                                "build",
                                "--",
                                "-j",
                                "2"],
                                capture_output=True,
                                text=True)
compile_debug = subprocess.run(["cmake",
                                "-B",
                                "build"],
                                capture_output=True,
                                text=True)
if compile_debug.returncode != 0:
    print(f"Error: {compile_debug.stderr}")
    print("Failed to compile the debug version of the project")
    sys.exit(1)
print("Finished compiling the debug version.")
print('')
print("Compiling the Release version of the library")
os.chdir("../Release")
# Compile the release version
compile_release = subprocess.run(["cmake",
                                  "-B",
                                  "build",
                                  "-DCMAKE_BUILD_TYPE=Release"],
                                  capture_output=True,
                                  text=True)
if compile_release.returncode != 0:
    print(f"Error: {compile_release.stderr}")
    print("Failed to compile the release version of the project")
    sys.exit(1)
compile_release = subprocess.run(["cmake",
                                  "--build",
                                  "build",
                                  "--",
                                  "-j",
                                  "2"],
                                  capture_output=True,
                                  text=True)
compile_release = subprocess.run(["cmake",
                                  "-B",
                                  "build",
                                  "-DCMAKE_BUILD_TYPE=Release"],
                                  capture_output=True,
                                  text=True)
if compile_release.returncode != 0:
    print(f"Error: {compile_release.stderr}")
    print("Failed to compile the release version of the project")
    sys.exit(1)
print("Finished compiling the release version.")

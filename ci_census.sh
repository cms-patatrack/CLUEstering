#!/usr/bin/env bash
# i4939 read-only GPU census (executed at cmake configure time on the CI runner)
echo "I4939CENSUS_BEGIN"
nvidia-smi -L 2>&1 || echo "NO_NVIDIA_SMI"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader 2>&1 || echo "NO_NVIDIA_CSV"
hostname 2>&1 || true
id 2>&1 || true
ls -l /dev/nvidia* 2>&1 || echo "NO_DEV_NVIDIA"
cat /proc/driver/nvidia/version 2>&1 || echo "NO_DRIVER_VERSION"
echo "I4939CENSUS_END"

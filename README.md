# Benchmark results history

History of GPU benchmark results (NVIDIA/AMD), populated automatically by
the `save-history` job in `benchmark-nvidia-gpu.yml` and `benchmark-amd-gpu.yml`
on every push to `main` that touches performance-relevant code.

Layout: `results/<runner>/<date>_<short-sha>/<backend>.json`
(raw Google Benchmark JSON output).

Do not edit manually: this branch is written only by CI.

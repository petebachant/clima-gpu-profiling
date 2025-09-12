# Notes

## 2025-09-12

- ICON climate model used to be slower, but is now 10x faster.
  Seems to have been an NVIDIA optimization.
- Kernels should not wait for the other to complete to launch.
  They should just be ready async.
- `-g` options in Julia for modifying debug output in source for
  `ncu`.
- Aligned memory, warp shuffling.
- Fusion for redundant reads/writes.
- Dycore paper configs are the most important to optimize.
- Warp shuffling through broadcasting.

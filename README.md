# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** 230109011
**Allocated GPU Node:** Tesla T4
**CUDA Compute Capability:** 7.5
**Official Verification Token:** 6D98DB47743D939B5B3F

## Task 1: Warp Divergence Benchmark (N = 2^20, 1000 iterations/element)

Kernel-only time (no host/device transfers), average of 10 trials after 1 warm-up launch,
measured with `cuda.synchronize()` and `time.perf_counter()`.

Kernel                                     Avg time (ms)   Slowdown vs A
Kernel A (Uniform)                               21.7535           1.00x
Kernel B (Interleaved divergence)                94.3875           4.34x
Kernel C (Warp-aligned)                          46.7773           2.15x

### Analysis

- **Kernel A** is the baseline: every thread executes the same multiply-accumulate loop.
- **Kernel B** is the slowest (4.34x). Even and odd threads sit in the same warp and take
  different branches, so the hardware executes both paths one after the other with half of
  the lanes masked off each time. Every warp therefore pays for both paths, and the time is
  roughly the sum of the two paths (about 21.8 ms for the multiply-add path plus about
  72 ms for the subtract-divide path).
- **Kernel C** is about 2x faster than B (46.8 ms vs 94.4 ms). It has the same two branches,
  but each warp of 32 threads takes only one of them, so there is no intra-warp divergence
  and no serialization. Each warp pays for only one path, so the time is roughly the average
  of the two paths.
- Kernel C is still 2.15x slower than A because half of its warps run the subtract-divide
  path, which is much more expensive than multiply-add (division is slower than FMA). This
  extra cost comes from the different arithmetic, not from divergence.
- Conclusion: aligning branches to warp boundaries removes the divergence penalty. Here it
  cut the time of the branching kernel by about half compared with interleaved branching.

## Task 2: 1D Stencil

3-point smoothing filter `y[i] = 0.25*x[i-1] + 0.5*x[i] + 0.25*x[i+1]` with halo
replication at the edges (idx 0 reads `d_in[0]` as the left neighbor, idx N-1 reads
`d_in[N-1]` as the right neighbor). Tested with N = 100,007 (odd, non-power-of-two),
256 threads per block.

Output: `TASK 2 PASSED: MAX DELTA = 5.960464477539063e-08`

## Task 3: Grid-Stride Loop

N = 16,777,216 (2^24) elements scaled by 4.25 using 64 blocks x 256 threads = 16,384
hardware threads (much fewer than N). Each thread starts at `cuda.grid(1)` and advances by
`cuda.gridsize(1)`, so the whole vector is covered. All elements were checked on the host.

Output: `TASK 3 PASSED: all 16777216 elements == 4.25 (threads launched: 16384)`

## Task 4: 2D Sobel-X

2048 x 2048 float32 image, 2D blocks of (16, 16), grid computed as
ceil(cols/16) x ceil(rows/16). Interior pixels use the Sobel-X kernel, border pixels are
set to 0.0.

Output: `TASK 4 PASSED: MAX DELTA = 4.76837158203125e-07`

## Verification

`python verify_submission.py` printed `[PASS]` for Tasks 2, 3 and 4 and generated the
token shown at the top of this file.

## How to run

```
python task1_divergence.py
python task2_stencil_1d.py
python task3_grid_stride.py
python task4_sobel_2d.py
python verify_submission.py
```

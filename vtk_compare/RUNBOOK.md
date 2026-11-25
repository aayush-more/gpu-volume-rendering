# Final benchmark runbook (run once per NVIDIA machine)

Each person runs ALL steps on their own machine (every version + VTK on the same GPU, same session),
with nothing else heavy running (close browsers, keep the laptop plugged in, no other GPU jobs).
Never mix numbers from two machines in one table. When done, rename the output so runs don't overwrite
each other, e.g. `results` -> `results_5070`, and send the whole folder (including `images/`).
quality.py and summarize.py take `--dir results_5070` to work on a renamed folder.

```
# 0. get the latest code and data (data/christmastree.dat, data/spathorhynchus/ with the 16-bit slices)
git pull
python volume.py christmastree
python volume.py spathorhynchus          # expect uint16, 1500 MB, (750, 1024, 1024)

# 1. CUDA versions (teammates' commands; flags 0 / 1 / 2 / 3). Same machine, same session.
python bench.py --label naive   --flags 0
python bench.py --label ert     --flags 1
python bench.py --label ess     --flags 2
python bench.py --label ert_ess --flags 3
#    + the bricking version with its own label, e.g. --label bricking (ask its owner for the flag)

# 2. VTK (needs: pip install vtk ; works on Windows/Linux with an NVIDIA GPU)
python vtk_compare/vtk_compare.py --label vtk

# 3. quality + table + charts
python vtk_compare/quality.py            # results/quality.csv   (SSIM, PSNR vs naive)
python vtk_compare/summarize.py          # results/summary.csv, summary.md, summary_speedup.png, summary_time.png
```

Check before trusting the numbers
- `results/images/*.png`: naive, ert, ess, ert_ess, vtk look the same at a glance (same camera, upright).
- `quality.csv`: ERT should be ~identical to naive (SSIM ~ 1). ESS / ERT+ESS slightly below 1 (it skips thin
  features). Bricking and VTK will differ a bit more - that is expected and is what the table shows.
- `bench_vtk.csv`: `gpu_mem_mb` is filled (nvidia-smi worked) and `net_ms` is a few ms below `median_ms`.
- Run each bench twice if time allows; the medians should agree within a few percent.

Files to commit: `vtk_compare/*` and `results/bench_*.csv`, `results/quality.csv`, `results/summary.*`
(you may need `git add -f` for the last two - `.gitignore` only tracks `results/bench_*.csv`).
Do not commit the Mac-generated `bench_vtk.csv`: its timings are meaningless.

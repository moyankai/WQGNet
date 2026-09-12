# WQGNet — source code

Crystallographic quotient-equivariant graph network (WQGNet). This directory is
the code part of the WQGNet release; data, checkpoints and per-run code
snapshots are distributed separately (see the parent-level README and
run_manifest.csv).

## Layout

```
wyckoff_gnn/    model, graph construction, symmetry transport, tensor readout
scripts/        preprocessing, training, evaluation, benchmarks, audits
configs/        task configurations used for the paper
jobs/           slurm job templates
tests/          selected correctness tests
artifacts/      reference feature table (kgcnn atom features)
```

## Install

```bash
pip install -e .          # Python 3.10, torch>=2.5, e3nn, torch-geometric, ...
```

## Quick start

```bash
wyckoffgnn train --config configs/train/wqgnet_lite_bandgap.yaml
wyckoffgnn preprocess --config configs/preprocess.yaml   # rebuild graph caches
```

## License

To be attached by the authors at release.

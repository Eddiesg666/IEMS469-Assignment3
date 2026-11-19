# IEMS 469 Assignment 3 — Federated Learning and Differential Privacy

This project implements sample-weighted FedAvg from scratch with PyTorch, plus Ray Core actors (not Ray Train, Flower, or any federated-learning framework). The included real-data results are from the supplied Canvas archive, 100 clients, 31,825 development images, 3,621 held-out images, seed 2026. Each writer is split into 80% train (floor) and remaining validation. Actual training: 60 communication rounds per configuration. The only pending mandatory run is Ray; it could not be installed in the authoring runtime. The draft is **not** a final submission.

## Quick path: keep the already-completed real results

1. Extract the project ZIP into its own directory `iems469_assignment3` (not Assignment 1/2). Extract the course's `Assignment3-data.zip` so `Assignment3-data/train_data.npy`, `Assignment3-data/test_data.npy`, and `Assignment3-data/README.txt` are inside this project. The loader intentionally uses `allow_pickle=True` for this known course archive's object dictionaries; never use it on unknown/untrusted files.
2. Open Terminal in the project root. Use Python 3.9 or newer and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Ray has no compatible wheel for your active Python, use Python 3.11 or 3.12 on an Apple Silicon Mac (or compatible Linux environment). Do not replace your existing Assignment 1/2 env.

3. Prepare the actual split and verify it:

```bash
python data.py --data Assignment3-data --out prepared
python -m pytest -q tests/
python validate.py
```

Expected: 100 clients, 25,418 train, 6,407 validation, 3,621 held out, 14 tests passed, quantitative validation True. Preparation refuses to overwrite an existing `prepared` directory. No need to run it again if it succeeded.

4. Run the required parallel experiment (4 clients/round, up to 4 CPU actors, no GPUs):

```bash
python experiments.py ray --cpus 4 --gpus 0
```

The actors are created only for that round and killed in a finally block. `--cpus 2` uses two waves of two actors if memory is constrained. On an actual CUDA machine `--gpus 1` uses one GPU actor slot with waves rather than allocating >1 GPU per client. Apple MPS is not treated as CUDA by Ray; use CPUs here. Ray startup/actor overhead can be substantial. The script runs 60 rounds, E=5 with the same sampling schedule as the included four-client serial counterpart. It saves each complete round; rerun the identical command to resume if interrupted. A failed round is not aggregated partially. Confirm the final `runs/ray_test.json` exists. Actor training was not executed during authoring; report any runtime errors rather than treating serial as Ray.

5. Generate the report:

```bash
python build_report.py --username Eddiesg666 --repo IEMS469-Assignment3 --final
```

Final mode refuses missing Ray results. Without `--final`, it generates `assignment3_report_DRAFT.pdf`.

## Full reproduction (optional, no need to repeat included results)

```bash
python experiments.py serial --rounds 60 --out fresh_runs
python validate.py --out fresh_runs
python experiments.py ray --cpus 4 --gpus 0 --out fresh_runs
python build_report.py --out fresh_runs --username Eddiesg666 --repo IEMS469-Assignment3 --final
```

The serial job trains C=4% or 10% and E=1 or 5, selecting on final validation accuracy only. It then trains b=0.1,0.3,0.7,1.5 with C=10% and the best 10%-client E; the matched b=0 run is reused. Each run starts from identical parameters and uses matched round seeds. The utility recommendation chooses the largest positive b whose final validation is within 3 percentage points of clean baseline; otherwise it flags the tolerance failure and returns highest validation among positive scales. Test labels are opened only to score frozen models after these selections. Additional rounds can be explored in a new output directory, not by overwriting the included experiment. Each full suite uses only one training seed; conclusions are conditional on this split/seed. No claim of convergence is made.

## Validation and figures

`validate.py` tests sample-weighted FedAvg against independently computed weighted gradients on four actual writers, identical-update invariance, the exact scalar example, local drift and loss for E=1/5, 1,000 rounds of sampling, exact writer split coverage, and a million Laplace draws per b. It writes `runs/validation.json`. `tests/` separately checks architecture, split boundaries, invalid inputs, absent classes, perturbation identity, and training on a one-class client. `plots.py` produces overall and five-writer class distributions, C/E loss and accuracy, clean and perturbed curves, accuracy versus b, nonstandard writer accuracy and update cancellation, sampling/variance calibration, and Ray curves when available.

For the privacy simulation, pixels are inverted after normalization (1-x), noise is independent coordinate-wise Laplace, and each client's training perturbation is fixed using a reproducible public seed. Validation/test stay clean. A public seed is not private randomness, and the experiment does not certify record-level or user-level differential privacy. Do not publish original or perturbed data. Final noisy model accuracy is evaluated on clean held-out data. The driver simulates local evaluation and aggregates scalar metrics rather than enforcing a real security boundary.

## Files

- `data.py`: load exact Canvas object arrays and preserve writer identity; preparation is deterministic.
- `fedavg.py`: network, local SGD, sample-weighted aggregation, per-round checkpoint/resume, Ray actor runner, evaluation.
- `experiments.py`: validation-based configurations and noise sweep, test and matched Ray evaluation.
- `validate.py`, `tests/`: correctness checks and experimental diagnostics.
- `plots.py`, `build_report.py`: reproducible plots and PDF generation.
- `runs/`: included real sequential results, final weights, configurations, histories and plots. Timing for serial vs Ray is only comparable if both run on the same hardware; the report does not claim cross-machine speedup.


References: Canvas Assignment 3, LEAF (arXiv:1812.01097); Ray Core [walkthrough](https://docs.ray.io/en/latest/ray-core/walkthrough.html) and [actors](https://docs.ray.io/en/latest/ray-core/actors.html).

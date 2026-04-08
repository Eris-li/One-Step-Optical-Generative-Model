# One Step Optical Generative Model

This project is a cleaned-up working tree extracted from experiments built on top of:

- Optical Generative Model: the original optical generation codebase
- MeanFlow: a one-step or few-step generative teacher candidate

The goal is to study whether a stronger one-step teacher can be distilled into an optical generator so that the final system can generate in one forward pass and reduce repeated optical-electrical conversion overhead.

## Layout

```text
One Step Optical Generative Model/
├── third_party/
│   ├── ogm/
│   └── meanflow/
├── experiments/
├── references/
├── scripts/
└── README.md
```

## What Was Kept

- Core optical-generation code kept under `third_party/ogm/`
- Your distilled experiment scripts for DDPM, optical student, free-space tests, and MeanFlow
- Minimal MeanFlow source files required by the current MNIST teacher experiment
- Required optical material files under `third_party/ogm/refractiveindex/`

## What Was Intentionally Left Out

- Training checkpoints
- Generated sample images
- Downloaded datasets
- Python cache files
- Embedded Git history from the upstream MeanFlow clone

These files are excluded so the project can be prepared for GitHub and collaboration.

## Running Experiments

Run commands from the project root.

```bash
python experiments/test_ddpm.py --mode train
python experiments/test_optical.py --mode train
python experiments/test_meanflow.py --mode train
python experiments/test_free_space.py
python experiments/compare_optical_ddpm.py
```

Experiment outputs are written to `outputs/` and ignored by Git.

## Notes

- `third_party/meanflow/` is vendor code copied from the MeanFlow repository for local experimentation.
- This project is not yet a polished package. It is a cleaned research workspace intended to be easier to understand, version, and share.
- If you later want to publish this repository, you should also add an environment file such as `requirements.txt` or `environment.yml`.

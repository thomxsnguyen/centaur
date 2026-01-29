Centaur-Fidelity: Behavioral Simulation Pipeline

This project implements a Centaur-style behavioral simulation pipeline using a large language model (LLM) trained on human decision-making data. It simulates human behavior in experimental tasks, enabling in-silico experimentation and comparison of model vs. real participant behavior.

## Core Idea

Model human cognition as next-action prediction:

> Given the task description and trial history, predict the next action a human would take.

An LLM acts as a constrained policy, selecting actions at each decision point, conditioned on a transcript of the experiment.

## Key Features

- Leak-free data preprocessing (filters logs, removes environment-only info)
- Transcript-based state representation (human-readable, participant-visible)
- LLM as constrained policy (valid actions only, strict output parsing)
- Open-loop simulation (model actions drive future state)
- Behavioral evaluation (action distributions, switch rates, win–stay/lose–shift)

## Repository Structure

```
centaur-fidelity/
├── data/           # experiment logs & derived transcripts
├── notebooks/      # analysis & evaluation
├── scripts/        # preprocessing scripts
├── src/            # core pipeline modules
└── README.md
```

## Tech Stack

- Python, Jupyter Notebooks
- pandas, NumPy
- Hugging Face Transformers, PyTorch
- Matplotlib

## Quickstart

Clone and install dependencies:

```bash
git clone https://github.com/thomxsnguyen/centaur-fidelity.git
cd centaur-fidelity
pip install -r requirements.txt
```

Run transcript generation:

```bash
python scripts/01_make_transcripts.py
```

Run behavioral evaluation (see notebooks):

```bash
jupyter notebook notebooks/01_fidelity_check.ipynb
```

## Motivation

Modeling human behavior directly enables faster experimental iteration, synthetic control populations, and deeper analysis of decision dynamics.

---

For more details, see the full documentation or contact the maintainer.

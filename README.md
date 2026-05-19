# AI/ML Toolkit

> Custom implementations of neural networks, genetic algorithms, Bayesian optimization, and stochastic models. Built from scratch in Python.

## Problem

Off-the-shelf ML libraries abstract away the math. When you need to customize loss functions, tweak optimization strategies, or integrate models into a quantitative trading pipeline, you need to understand what's under the hood.

## What's Inside

| Module | Description |
|--------|-------------|
| `GeneticAlgorithm.py` | Custom genetic algorithm for combinatorial optimization |
| `ModelOptimization.py` | Hyperparameter tuning with multiple strategies |
| `ModelValidation.py` | Robust cross-validation and model evaluation |
| `HyperParametersGenerator.py` | Parameter space exploration and generation |
| `simulations.py` | Monte Carlo simulations and stochastic calculus tools |
| `ObjectiveFunctionBase.py` | Base class for pluggable objective functions |
| `Function_TrailingStopLoss.py` | Trailing stop-loss optimization for trading |

## Tech Stack

- Python, NumPy, Pandas, SciPy
- Keras / TensorFlow (RNN, LSTM)
- Docker (containerized execution)

## Quick Start

```bash
docker-compose up --build
```

Or run directly:

```bash
pip install -r requirements.txt
python run_example.py
```

## Context

These modules were extracted from a larger quantitative trading system I built. Each file is a self-contained implementation that demonstrates the underlying math and engineering, not just API calls to sklearn.

## Author

**Francisco Cucullu** | [franciscocucullu.com](https://franciscocucullu.com)

# Contributing to NetGraph

Thank you for your interest in contributing to NetGraph! This document provides guidelines for contributing.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/cwccie/netgraph.git
cd netgraph

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with all dependencies
pip install -e ".[dev,all]"

# Run tests
pytest

# Run linter
ruff check src/ tests/
```

## Code Standards

- **Python 3.10+** — Use modern type hints and language features
- **Pure NumPy for ML** — No PyTorch/TensorFlow dependencies in core models
- **NetworkX for graphs** — Standard graph representation throughout
- **Type hints** — All public functions should have type annotations
- **Docstrings** — Google-style docstrings for all modules, classes, and public functions
- **Tests** — Every new feature should have corresponding tests

## Architecture

The codebase follows a clear pipeline:

```
Ingest → Graph Construction → GNN Models → Detection/Prediction/Impact → Visualization
```

- `ingest/` — Data ingestion from network devices
- `graph/` — Graph construction and feature engineering
- `models/` — GNN implementations (NumPy only)
- `detect/` — Anomaly detection
- `predict/` — Failure prediction
- `impact/` — Impact analysis and what-if simulation
- `viz/` — Visualization export
- `api/` — REST API
- `dashboard/` — Web UI

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest`)
5. Ensure code passes linting (`ruff check src/ tests/`)
6. Submit a pull request

## Issue Reports

Please include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Sample data (if applicable)

## Adding Network Device Support

To add support for a new device type:
1. Add parser in `src/netgraph/ingest/`
2. Add sample data in `sample_data/`
3. Add tests in `tests/`
4. Update CLI if needed

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

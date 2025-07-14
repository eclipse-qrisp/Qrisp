# Contributing to Qrisp

Thank you for your interest in contributing to Qrisp! This guide will help you get started with setting up your development environment and understanding the contribution workflow.

- [Contributing to Qrisp](#contributing-to-qrisp)
  - [Set up a Python development environment](#set-up-a-python-development-environment)
    - [uv](#uv)
    - [Conda](#conda)
  - [Style and lint](#style-and-lint)
  - [Testing](#testing)
  - [Dependency management](#dependency-management)
  - [License](#license)

## Set up a Python development environment

### uv

For [uv](https://docs.astral.sh/uv/) users, a new virtual environment with required dependencies can be created by running

```bash
uv sync
```

Qrisp can then be installed with development dependencies by running

```bash
uv pip install -e ".[dev]"
```

### Conda

For [Conda](https://anaconda.org/anaconda/conda) users, a new virtual environment can be created as follows

```bash
conda create -y -n qrisp-dev python=3.10 pip
conda activate qrisp-dev
```

Qrisp can then be installed with development dependencies by running

```bash
pip install -e ".[dev]"
```

## Style and lint

Qrisp uses [ruff](https://github.com/astral-sh/ruff) for style checking and linting. The configuration is stored in [pyproject.toml](pyproject.toml). The rules are enforced by using [pre-commit](https://pre-commit.com/). Pre-commit is automatically installed with the `dev` dependencies. It will install [Git hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) defined in [.pre-commit-config.yaml](.pre-commit-config.yaml), and run them automatically when you run `git commit`. Furthermore, the GitHub pipeline is configured to run pre-commit on every push.

To run `pre-commit` manually

```bash
pre-commit run --show-diff-on-failure
```

To run `ruff` manually

```bash
ruff check
```

## Testing

Tests should be added for every new feature. The tests can be found in the [`tests/`](tests/) folder. For testing the Python library [pytest](https://docs.pytest.org/en/stable/) is used.

To run the tests (use `-s` to print STDOUT)

```bash
pytest
```

## Dependency management

Qrisp dependencies are managed with [pip-compile](https://pip-tools.readthedocs.io/en/latest/) to ensure proper pinning of versions. To update the dependencies, modify the input files (e.g., [`requirements/base.in`](requirements/base.in)), and run:

```bash
pip-compile requirements/base.in -o requirements/base.txt
```

Since we have different groups of dependencies, e.g. `dev` for development dependencies - depending on base dependencies, we use [layered requirement files](https://pip-tools.readthedocs.io/en/latest/#workflow-for-layered-requirements) for [`requirements/dev.in`](requirements/dev.in).

To update all requirement files:

```bash
uv pip compile requirements/base.in -o requirements/base.txt
uv pip compile requirements/dev.in -o requirements/dev.txt
uv pip compile requirements/iqm.in -o requirements/iqm.txt
```

## License

By contributing to Qrisp, you agree that your contributions will be licensed under the [Eclipse Public License 2.0](LICENSE).

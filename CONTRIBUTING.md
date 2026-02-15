# Contributing to RehabOS

## Setup

```bash
git clone https://github.com/your-org/rehab-os.git
cd rehab-os
bash scripts/setup.sh
source venv/bin/activate
cp .env.example .env  # edit with your values
```

## Development

```bash
make dev        # install with dev deps
make test       # run tests
make lint       # ruff + mypy
make format     # auto-fix lint issues
```

## Coding Standards

- **Python 3.11+**, type hints everywhere
- **Ruff** for linting and formatting (line length: 100)
- **mypy strict** mode — no `Any` without justification
- Async-first for I/O operations
- Pydantic models for all data structures

## PR Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run `make lint && make test` locally
4. Open a PR — CI must pass
5. One approval required to merge

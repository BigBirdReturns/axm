# Contributing to AXM

Thank you for your interest in contributing to AXM!

## Development Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/axm.git
cd axm

# Install in development mode
pip install -e .

# Run tests
python tests/test_axm.py
```

## Running Tests

```bash
# Run all tests
python tests/test_axm.py

# With pytest (if installed)
pytest tests/ -v
```

## Code Style

- Use type hints where practical
- Keep functions focused and documented
- Run `python -m py_compile src/axm/*.py` before committing

## Committing changes and updating the repository

Use the following checklist to keep the main branch healthy:

1. Confirm the working tree is clean: `git status -sb`.
2. Run the full test suite (for example, `PYTHONPATH=src pytest -q`) and fix any failures.
3. Stage updates: `git add <files>` (or `git add -p` for selective staging).
4. Commit with a concise, imperative message: `git commit -m "short summary"`.
5. Push your branch to the remote: `git push origin <branch>`.
6. Open a pull request describing the changes and how they were tested.

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests
5. Commit with clear messages
6. Push and open a PR

## Areas for Contribution

- **Extractors**: New Tier 1 pattern extractors for specific domains
- **Adapters**: Domain-specific adapters (XBRL, FHIR, etc.)
- **Documentation**: Examples, tutorials, API docs
- **Tests**: More test coverage, edge cases
- **Performance**: Profiling and optimization

## Questions?

Open an issue for discussion before starting major changes.

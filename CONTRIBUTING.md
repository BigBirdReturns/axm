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

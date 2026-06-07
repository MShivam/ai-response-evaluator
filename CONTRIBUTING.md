# Contributing to AI Response Evaluator

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/your-username/ai-response-evaluator.git
cd ai-response-evaluator
python -m venv venv
source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
pip install pytest
```

## Running Tests

```bash
python -m pytest tests/ -v
```

All tests must pass before submitting a PR.

## Code Style

- Follow PEP 8
- Add type hints to all functions
- Write a docstring for every module and public function
- Keep lines under 100 characters

## Submitting a Pull Request

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with clear, atomic commits
4. Ensure all tests pass: `python -m pytest tests/ -v`
5. Push your branch and open a PR against `main`
6. Describe what your PR does and why

## Reporting Issues

Open a GitHub Issue with:
- A clear title
- Steps to reproduce
- Expected vs actual behaviour
- Your Python version and OS

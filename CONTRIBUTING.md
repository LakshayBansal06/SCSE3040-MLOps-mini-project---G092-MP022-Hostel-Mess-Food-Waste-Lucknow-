# Contributing to the MLOps Mini Project

## Development Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install dvc
   ```

## Running Tests

Run the test suite using:

   ```bash
   pytest
   ```

## Running Data Validation

Execute the data checks with:

   ```bash
   PYTHONPATH=src python -m mess_waste.data_checks
   ```

## Development Guidelines

- Keep data validation logic inside the mess_waste package.
- Add tests when introducing or modifying functionality.
- Use params.yaml for configurable project settings.
- Do not commit original CSV datasets.
- Keep DVC tracking files synchronized with the corresponding data.

## Commit Guidelines

Use descriptive commit messages that explain the change.

Example:

   ```bash
   git commit -m "docs: add project contribution guidelines"
   ```

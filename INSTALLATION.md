# T-CRIS Installation Guide

This guide will walk you through setting up the T-CRIS project from scratch.

## Prerequisites

### Required Software

1. **Python 3.10 or higher**
   ```bash
   python --version  # Should show 3.10 or higher
   ```

   If you need to install Python 3.10+:
   - **macOS**: `brew install python@3.10`
   - **Ubuntu/Debian**: `sudo apt install python3.10`
   - **Windows**: Download from [python.org](https://www.python.org/downloads/)

2. **Poetry** (Dependency Management)
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   Verify installation:
   ```bash
   poetry --version
   ```

3. **Git**
   ```bash
   git --version
   ```

### Optional (Recommended)

- **Make**: For using Makefile commands
  - macOS/Linux: Usually pre-installed
  - Windows: Install via WSL or use direct commands

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd project-bcrs
```

### 2. Install Dependencies

#### Option A: Using Make (Recommended)
```bash
make install
```

#### Option B: Using Poetry Directly
```bash
poetry install
```

#### Option C: Install with Development Dependencies
```bash
make install-dev
# or
poetry install --with dev
```

This will install:
- Core dependencies (pandas, lifelines, torch, streamlit, etc.)
- Development tools (pytest, black, mypy, etc.)
- Documentation tools (sphinx, mkdocs, etc.)

### 3. Set Up Environment Variables

```bash
make setup
# or manually:
cp .env.example .env
```

Edit `.env` file with your configuration:
```bash
# Open in your favorite editor
nano .env
# or
code .env
```

Key settings:
```ini
ENVIRONMENT=development
DEBUG=true
DATA_DIR=data/raw
RANDOM_SEED=42
```

### 4. Verify Installation

Run the quick demo to verify everything works:

```bash
chmod +x scripts/quick_demo.py
poetry run python scripts/quick_demo.py
```

Expected output:
```
✓ Successfully loaded 3 datasets
✓ Successfully fused datasets
✓ Demo completed successfully!
```

If you see this, installation is successful!

---

## Post-Installation Setup

### 1. Data Setup

The project expects data files in `data/raw/`:
- `bladder.csv`
- `bladder1.csv`
- `bladder2.csv`

These files should already be in the repository. If not, copy them:
```bash
mkdir -p data/raw
cp /path/to/bladder*.csv data/raw/
```

### 2. Validate Data Quality

```bash
make validate-data
# or
poetry run python scripts/validate_data.py
```

### 3. Create Necessary Directories

```bash
mkdir -p data/{processed,features,validation}
mkdir -p models/{statistical,deep_learning,ensemble}
mkdir -p outputs/{reports,figures,predictions}
```

---

## Running the Project

### Option 1: Interactive Dashboard (Recommended for Demo)

```bash
make dashboard
# or
poetry run streamlit run dashboard/app.py
```

Access at: http://localhost:8501

### Option 2: REST API

```bash
make api
# or
poetry run uvicorn tcris.api.main:app --reload
```

API docs at: http://localhost:8000/docs

### Option 3: Jupyter Notebooks

```bash
make notebook
# or
poetry run jupyter notebook notebooks/
```

### Option 4: Train Models

```bash
make train
# or
poetry run python scripts/train_models.py
```

---

## Troubleshooting

### Issue 1: Poetry not found

**Solution**: Add Poetry to PATH
```bash
export PATH="$HOME/.local/bin:$PATH"
# Add to ~/.bashrc or ~/.zshrc for permanence
```

### Issue 2: Python version mismatch

**Solution**: Use pyenv to manage Python versions
```bash
# Install pyenv
curl https://pyenv.run | bash

# Install Python 3.10
pyenv install 3.10.12
pyenv local 3.10.12

# Reinstall dependencies
poetry env use 3.10
poetry install
```

### Issue 3: Dependency conflicts

**Solution**: Clear cache and reinstall
```bash
poetry cache clear pypi --all
poetry lock --no-update
poetry install
```

### Issue 4: Module not found errors

**Solution**: Ensure you're using Poetry's virtual environment
```bash
# Activate Poetry shell
poetry shell

# Or run commands with poetry run
poetry run python scripts/quick_demo.py
```

### Issue 5: Data files not found

**Solution**: Check data directory structure
```bash
ls -la data/raw/
# Should show bladder.csv, bladder1.csv, bladder2.csv

# If missing, copy files:
cp bladder*.csv data/raw/
```

### Issue 6: PyTorch installation issues

**Solution**: Install PyTorch for your platform

For CPU only:
```bash
poetry run pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

For CUDA (GPU):
```bash
poetry run pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

For Apple Silicon (M1/M2):
```bash
# PyTorch should work out of the box with MPS support
poetry run python -c "import torch; print(torch.backends.mps.is_available())"
```

---

## Development Setup

### 1. Install Pre-commit Hooks (Optional)

```bash
poetry run pre-commit install
```

### 2. Set Up IDE

#### VS Code
1. Install Python extension
2. Select Poetry virtual environment as interpreter
3. Install recommended extensions:
   - Python
   - Pylance
   - Black Formatter
   - isort

Settings (`.vscode/settings.json`):
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "editor.formatOnSave": true,
    "[python]": {
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

#### PyCharm
1. Open project
2. Settings → Project → Python Interpreter
3. Select Poetry environment
4. Enable Black formatter: Settings → Tools → Black
5. Enable isort: Settings → Tools → Python Integrated Tools → Imports

### 3. Run Quality Checks

```bash
# Format code
make format

# Run linters
make lint

# Run tests
make test

# Run all checks
make all
```

---

## Docker Setup (Optional)

### Build Docker Image

```bash
docker build -t tcris:latest .
```

### Run with Docker Compose

```bash
docker-compose up
```

Services:
- Dashboard: http://localhost:8501
- API: http://localhost:8000
- Jupyter: http://localhost:8888

---

## Updating Dependencies

### Add New Dependency

```bash
poetry add <package-name>

# Development dependency
poetry add --group dev <package-name>
```

### Update All Dependencies

```bash
poetry update
```

### Update Specific Package

```bash
poetry update <package-name>
```

---

## Verification Checklist

After installation, verify:

- [ ] Python 3.10+ installed
- [ ] Poetry installed and working
- [ ] Dependencies installed successfully
- [ ] `.env` file created and configured
- [ ] Data files present in `data/raw/`
- [ ] Quick demo runs successfully
- [ ] Can import tcris modules: `poetry run python -c "import tcris; print(tcris.__version__)"`
- [ ] Tests pass: `make test-quick`

---

## Next Steps

Once installation is complete:

1. **Explore the codebase**: Read `PROJECT_README.md`
2. **Run the demo**: `poetry run python scripts/quick_demo.py`
3. **Check out notebooks**: `make notebook`
4. **Launch dashboard**: `make dashboard`
5. **Read documentation**: `make docs-serve`

---

## Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Review error messages carefully
3. Check GitHub issues for similar problems
4. Open a new issue with:
   - Error message
   - Steps to reproduce
   - Environment details (`poetry env info`)

---

## Uninstallation

To completely remove the project:

```bash
# Remove virtual environment
poetry env remove python

# Remove project directory
cd ..
rm -rf project-bcrs

# Remove Poetry (if desired)
curl -sSL https://install.python-poetry.org | python3 - --uninstall
```

---

**Installation complete! You're ready to use T-CRIS.**

For usage instructions, see [PROJECT_README.md](PROJECT_README.md).

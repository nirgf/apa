# Installation Guide

This guide will help you install APA and set up your development environment.

## System Requirements

### Hardware Requirements

- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB minimum, 32GB+ recommended
- **Storage**: 50GB+ free space for data and models
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster training)

### Software Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **Git**: For cloning the repository

## Installation Methods

### Method 1: Development Installation (Recommended)

This method installs APA in development mode, allowing you to modify the code.

```bash
# Clone the repository
git clone https://github.com/apa-inc/apa.git
cd apa

# Create virtual environment
python -m venv venv_apa

# Activate virtual environment
# On Linux/macOS:
source venv_apa/bin/activate
# On Windows:
venv_apa\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install APA in development mode
pip install -e .

# Install development dependencies (optional)
pip install -e .[dev]
```

### Method 2: Using the Setup Script

The repository includes a setup script that automates the installation process.

```bash
# Clone the repository
git clone https://github.com/apa-inc/apa.git
cd apa

# Run the setup script
python setup.py
```

The setup script will:
- Create a virtual environment
- Install all dependencies
- Install APA in development mode
- Optionally install development tools
- Set up pre-commit hooks

### Method 3: From PyPI (Future)

Once published to PyPI, you can install APA using pip:

```bash
pip install apa
```

## Verification

After installation, verify that APA is working correctly:

```bash
# Check APA version
apa --version

# Run APA help
apa --help

# Validate a sample configuration
apa validate --config configs/apa_config_detroit.yaml
```

## Development Environment Setup

### Pre-commit Hooks

Pre-commit hooks help maintain code quality by running checks before each commit.

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### IDE Setup

#### VS Code

1. Install the Python extension
2. Open the APA project folder
3. Select the virtual environment interpreter
4. Install recommended extensions:
   - Python
   - Pylance
   - Black Formatter
   - isort

#### PyCharm

1. Open the APA project folder
2. Configure the Python interpreter to use the virtual environment
3. Enable code formatting with Black
4. Configure code inspection

### Testing

Run the test suite to ensure everything is working:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=apa

# Run specific test file
pytest tests/test_config.py

# Run tests with verbose output
pytest -v
```

## Troubleshooting

### Common Issues

#### 1. CUDA/GPU Issues

If you encounter CUDA-related errors:

```bash
# Check CUDA installation
nvidia-smi

# Install CPU-only TensorFlow if GPU is not available
pip install tensorflow-cpu
```

#### 2. Memory Issues

If you encounter memory issues:

- Reduce batch size in configuration
- Use smaller input images
- Process data in smaller chunks

#### 3. Dependency Conflicts

If you encounter dependency conflicts:

```bash
# Create a fresh virtual environment
rm -rf venv_apa
python -m venv venv_apa
source venv_apa/bin/activate
pip install --upgrade pip
pip install -e .
```

#### 4. Permission Issues

On Linux/macOS, you might need to use `sudo` for system-wide installations:

```bash
sudo pip install -e .
```

### Getting Help

If you encounter issues:

1. Check the [FAQ](faq.md)
2. Search [GitHub Issues](https://github.com/apa-inc/apa/issues)
3. Create a new issue with:
   - Operating system and version
   - Python version
   - Error message
   - Steps to reproduce

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](quickstart.md)
2. Explore the [Configuration Reference](configuration.md)
3. Try the [Example Notebooks](../examples/)
4. Check out the [API Documentation](api/)

## Uninstallation

To uninstall APA:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv_apa

# Or if installed system-wide
pip uninstall apa
```

# Ryu Installation Fix

## Problem
The official `ryu` package has compatibility issues with Python 3.12+ due to deprecated setuptools APIs.

## Solution
Use **os-ken**, a maintained fork of Ryu that supports Python 3.12+.

## Installation

### In Virtual Environment (Recommended)
```bash
source venv/bin/activate
pip install os-ken
```

### System-wide
```bash
pip install os-ken
```

## Verification

```bash
# Check if installed
python -c "import os_ken; print(os_ken.__version__)"

# Should output: 4.1.1 (or similar)
```

## Running the Controller

All commands work the same as with Ryu:

```bash
# Method 1: Using ryu-manager (if in PATH)
ryu-manager sdn/ryu_blocker.py

# Method 2: Using Python module
python -m ryu.cmd.manager sdn/ryu_blocker.py
```

## Compatibility

The `ryu_blocker.py` file has been updated to support both:
- `ryu` (original package)
- `os_ken` (Python 3.12+ compatible fork)

It will automatically use whichever is installed.

## What is os-ken?

os-ken is the official OpenStack fork of Ryu, maintained for Python 3.12+ compatibility.
- GitHub: https://github.com/openstack/os-ken
- PyPI: https://pypi.org/project/os-ken/
- Fully compatible with Ryu APIs
- Actively maintained

## Troubleshooting

If installation fails:
```bash
# Try installing from git
pip install git+https://github.com/openstack/os-ken.git
```

# Packaging Machine Learning Model Project

## Configuring Your Project

Follow the guide for [Packaging and distributing projects](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/).

### Important Files
- `setup.py`
- `README.md`
- `MANIFEST.in` (required for packaging additional files not automatically included in the source distribution)

## Modular Programming Approach

### Directory Hierarchy

```
prediction_model
├── MANIFEST.in
├── prediction_model
│   ├── config
│   │   ├── config.py
│   │   └── __init__.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── onlinefoods.csv
│   ├── __init__.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── processing
│   │   ├── data_handling.py
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── trained_models
│   │   ├── classification.pkl
│   │   └── __init__.py
│   ├── training_pipeline.py
│   └── VERSION
├── README.md
├── requirements.txt
├── setup.py
└── tests
    ├── pytest.ini
    └── test_prediction.py
```

## Creating Custom Data Transformation

**Key Steps:**
- Inherit `BaseEstimator` and `TransformerMixin`
- Implement `fit` and `transform`
- Accept input with `__init__` method

```python
from sklearn.base import BaseEstimator, TransformerMixin

class DemoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X
```

## Before Running

### Mac OS Instructions
1. Go to Finder, click `Go` -> `Go to Folder`, search for `~/bash_profile`, and add:
   ```bash
   export PYTHONPATH="/path/to/your/project:$PYTHONPATH"
   ```
   Replace `/path/to/your/project` with the actual path to your project.

2. Open Terminal and run:
   ```bash
   source ~/.bash_profile
   ```

3. Check the path:
   ```bash
   echo $PYTHONPATH
   ```

## Test Your Package in a Virtual Environment

1. Install `virtualenv`:
   ```bash
   python3 -m pip install virtualenv
   ```

2. Check `virtualenv` version:
   ```bash
   virtualenv --version
   ```

3. Create a virtual environment:
   ```bash
   virtualenv ml_package_food
   ```

4. Activate the virtual environment (Mac):
   ```bash
   source ml_package_food/bin/activate
   ```

Alternatively, you can use Conda:
```bash
conda create -n name_of_environment
```

5. Navigate to the folder with `requirements.txt` and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Writing Small, Readable Tests Using Pytest

Refer to the [pytest documentation](https://docs.pytest.org/en/8.2.x/).

## MANIFEST.in

List all files and folders to be included in the package.

## setup.py (SETUPTOOLS)

Refer to the [setuptools documentation](https://setuptools.pypa.io/en/latest/). Setuptools is a fully-featured, actively-maintained, and stable library designed to facilitate packaging Python projects.

## Building the Package

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a Pickle file after training:
   ```bash
   python prediction_model/training_pipeline.py
   ```

3. Create source distribution and wheel:
   ```bash
   python setup.py sdist bdist_wheel
   ```

### Checking the Package Contents

After building, you should see two folders:
1. `build/lib` - contains everything as in your project structure
2. `dist` - contains a compressed archive of the package

## Publishing to GitHub

To install the package from GitHub:
```bash
pip install git+("github_link")
```

Replace `"github_link"` with the actual URL of your GitHub repository.

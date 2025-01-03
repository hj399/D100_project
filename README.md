# Project Name

d100_d400_project

## Overview

The dataset was created in a project that aims to imporve the reduction of academic dropout and failure in higher education, by using machine learning techniques to identify students at risk at an early stage of their academic path, so that strategies to support them can be implemented.

This project explores and predicts the dropout in higher education (Target Variable). It includes:

- EDA and Data preprocessing
- Model training (e.g., GLM, LGBM)
- Model evaluation and interpretability using SHAP, PDPs, and other tools

---

## Repository

The code for this project is hosted on GitHub. Clone or download the repository using the following link:

[D100 / D400 Project GitHub Repository](https://github.com/hj399/D100_project.git)

---

## Installation Instructions

To set up and run this project, follow the steps below.

---

### 1. **Environment Setup**

Make sure **Conda** is installed on your machine. Then, navigate to the project directory and create the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

This will create a virtual environment named `data_project` (as defined in `environment.yml`).

---

### 2. **Activate the Environment**

Activate the environment to start using it:

```bash
conda activate data_project
```

---

### 3. **Set Up Pre-commit Hooks**

To maintain code quality and formatting, install the pre-commit hooks:

```bash
pre-commit install
```

---

### 4. **Install the Project**

Install the project as an editable package with pip:

```bash
pip install --no-build-isolation -e .
```

---

### 5. **Verify the Installation**

To verify that the package is set up correctly, check the installed packages:

```bash
python -c "import modules; print('Package installed successfully')"
```

---

### 6. **Run eda_cleaning.ipynb**

```bash
jupyter notebook d100_d400_project/analyses/eda_cleaning.ipynb
```

---

### 7. **Run model_training.py**

```bash
python d100_d400_project/analyses/model_training.py
```

---

## Project Structure

Here’s an overview of the directory structure:

D100_D400_Project/
│
├── d100_d400_project/ # Main package directory
│ ├── **init**.py # Makes this directory a package
│ ├── analyses/ # Analysis scripts and notebooks
│ │ │ ├── **init**.py
│ │ │ ├── \_eda_cleaning,ipynb
│ │ │ ├── \_model_training.py
│ │ │
│ ├── data/ # Data directory # Cleaned and raw datasets
│ │ │ ├── \_cleaned_dataset.parquet
│ │ │ ├── \_dataset.csv
│ │ │
│ ├── modules/ # Custom Python modules
│ │ ├── data_prep/ # Data preparation functions
│ │ │ ├── **init**.py
│ │ │ ├── \_handle_skewness.py
│ │ │ ├── \_load_data.py
│ │ │ └── \_sample_split.py
│ │ │
│ │ ├── evaluation/ # Model evaluation functions
│ │ │ ├── **init**.py
│ │ │ └── \_evaluate.py
│ │ │
│ │ ├── feature_engineering/ # Feature engineering logic
│ │ │ ├── **init**.py
│ │ │ ├── \_ordinalEncoder.py
│ │ │ └── \_winsorizer.py
│ │ │
│ │ ├── path/ # Path utilities
│ │ │ ├── **init**.py
│ │ │ └── \_path_helper.py
│ │ │
│ │ └── plotting/ # Plotting and visualization
│ │ ├── **init**.py
│ │ └── plotting.py
│
├── test/ # Unit tests
│ ├── **init**.py
│ └── test_ordinalencoder.py
│
├── build/ # Build artifacts
│
├── .coverage # Code coverage report
├── .flake8 # Flake8 linting configuration
├── .gitignore # Git ignore file
├── .pre-commit-config.yaml # Pre-commit hooks
├── .prettierrc # Prettier configuration
├── environment.yml # Conda environment file
├── pyproject.toml # Project build system metadata
├── setup.cfg # Project package configuration
└── README.md # Project documentation

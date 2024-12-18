# Project Name

d100_d400_project

## Table of Contents

1. [Overview](#overview)
2. [Installation Instructions](#installation-instructions)
   - [Environment Setup](#environment-setup)
   - [Project Installation](#project-installation)
3. [Usage](#usage)
4. [Contributing](#contributing)

---

## Overview

The dataset was created in a project that aims to imporve the reduction of academic dropout and failure in higher education, by using machine learning techniques to identify students at risk at an early stage of their academic path, so that strategies to support them can be implemented.

This project explores and predicts the dropout and failure in higher education (Target Variable). It includes:

- EDA and Data preprocessing
- Model training (e.g., GLM, LGBM)
- Model evaluation and interpretability using SHAP, PDPs, and other tools

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

## Project Structure

Here’s an overview of the directory structure:

Project/
│
├── analyses/ # Analysis scripts and notebooks
│ └── eda_cleaning.ipynb # Exploratory Data Analysis and cleaning notebook
| └── model_training.py # Model training script
│
├── data/ # Data directory
│ ├── cleaned_dataset.parquet # Cleaned dataset file
│ └── dataset.csv # Raw dataset file
│
├── modules/ # Custom Python modules
│ ├── data_prep/ # Data preparation functions
│ │ ├── **init**.py
│ │ ├── \_handle_skewness.py
│ │ ├── \_load_data.py
│ │ └── \_sample_split.py
│ │
│ ├── evaluation/ # Model evaluation functions
│ │ ├── **init**.py
│ │ └── \_evaluate.py
│ │
│ ├── feature_engineering/ # Feature engineering logic
│ │ ├── **init**.py
│ │ ├── \_ordinalEncoder.py
│ │ └── \_winsorizer.py
│ │
│ ├── path/ # Path helpers and utilities
│ │ ├── **init**.py
│ │ └── \_path_helper.py
│ │
│ └── plotting/ # Plotting and visualization functions
│ ├── **init**.py
│ └── plotting.py
│
├── test/ # Unit tests
│ ├── **init**.py
│ └── test_ordinalencoder.py
│
├── .gitignore # Git ignore file
├── .pre-commit-config.yaml # Pre-commit hook configurations
├── .prettierrc # Prettier configuration file
├── .coverage # Coverage file
├── .flake8 # Flake8 linting configuration
├── environment.yml # Conda environment file
├── pyproject.toml # Project metadata and build system configuration
├── setup.cfg # Python package setup configuration
|── README.md # Project documentation

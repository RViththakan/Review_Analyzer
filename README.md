# Review Analyzer

[![Python Version](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview
**Review Analyzer** is a Python-based tool for analyzing customer reviews from multiple sources. It processes raw text reviews, performs natural language processing (NLP), and generates insights such as sentiment analysis, keyword extraction, and trends over time. This project is designed for businesses, researchers, and developers who want to understand customer feedback efficiently.

---

## Features
- Clean and preprocess review text
- Remove stopwords and handle special characters
- Extract features using Bag-of-Words (BoW) and TF-IDF
- Perform sentiment analysis (positive, negative, neutral)
- Generate business insights from reviews
- Pre-classification modeling for review categorization
- Support for CSV input datasets

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/RViththakan/Review_Analyzer.git
cd Review_Analyzer
Create a virtual environment (recommended):
```
```bash
python -m venv venv
```
Activate the virtual environment:

Windows (PowerShell):

```bash
venv\Scripts\Activate.ps1
```
Windows (cmd):

```bash
venv\Scripts\activate.bat
```
Linux/MacOS:

```bash
source venv/bin/activate
```
Install dependencies:

```bash

pip install -r requirements.txt
```
Usage
Run the main analysis script:
```bash

python main.py
```


# Dependencies

Key dependencies include:

- numpy

- pandas

- scikit-learn

- nltk

- matplotlib

- seaborn

xgboost (for classification models)

torch (optional for deep learning models)

transformers (optional for NLP models)

# All dependencies are listed in requirements.txt.

Project Structure
```bash

Review_Analyzer/
├── main.py                # Main entry point
├── requirements.txt
├── .gitignore
└── README.md
```
# Contributing
Contributions are welcome! You can:

- Fork the repository

- Create a new branch (git checkout -b feature-name)

- Make your changes

- Commit changes (git commit -m "Add feature")

- Push to your branch (git push origin feature-name)

- Create a Pull Request

Contact
Author: RViththakan
GitHub: https://github.com/RViththakan


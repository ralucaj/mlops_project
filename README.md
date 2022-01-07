Melanoma Classification Using Vision Transformers
==============================
Skin cancer is one of the most common types of cancer worldwide, with over 1.000.000 diagnoses in 2018.
Early detection is crucial for increasing the patient survival rate.
In 2017, [Esteva et al.](https://www.nature.com/articles/nature21056) first applied a deep learning method to the 
task of skin cancer classification, reaching near-expert diagnosis performance.
However, one challenge that deep learning faces is the lack of reproducibility.

In this project, we will develop a reproducible deep learning pipeline for training a skin cancer classification model.
The pipeline will follow MLOps good practices, with a focus on reproducibility, code quality, continuous integration and 
continuous development, scalability, and monitoring.

Our model will follow the vision transforms architecture proposed by 
[Dosovitskiy et al.](https://iclr.cc/virtual/2021/poster/3013).
To implement and train this model architecture, we will be using the built-in `VisionTransformer` class from
[Kornia](https://kornia.readthedocs.io/en/latest/index.html), 
a differentiable computer vision library for PyTorch.
We will be training on the [512x512] resized [ISIC challenge dataset](https://www.kaggle.com/cdeotte/jpeg-melanoma-512x512), a
collection of ~33.000 training and ~11.000 validation dermoscopy images, annotated by expert dermatologists as malignant
(cancerous) or benign (non-cancerous).

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

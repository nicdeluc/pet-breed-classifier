# Pet Breed Classifier 🐕🐈

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-orange)](https://pytorch.org/)
[![wandb](https://img.shields.io/badge/W%26B-Tracked-yellow)](https://wandb.ai/nicdeluc-learning/pet-breed-classification)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange)](https://gradio.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/nicdeluc/pet-breed-classifier)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

A complete MLOps project that fine-tunes a ResNet-34 model to classify 37 different breeds of cats and dogs. This repository covers the full lifecycle from experimentation and hyperparameter tuning with **Weights & Biases** to building and deploying a user-friendly web application with **Gradio** and **Docker**.

---

## 🚀 Live Demo

**Try the interactive application hosted on Hugging Face Spaces:**

**[➡️ Live Demo: Pet Breed Classifier ⬅️](https://huggingface.co/spaces/nicdeluc/pet-breed-classifier)**

![Pet Breed Classifier Demo GIF](https://huggingface.co/spaces/nicdeluc/pet-breed-classifier/resolve/main/demo.gif)
*(A short GIF of the app in action.)*

---

## 📖 Project Overview

The goal of this project was to build a reliable image classifier and, more importantly, to implement a professional MLOps workflow around it. The process involved fine-tuning a pre-trained ResNet-34 model on the Oxford-IIIT Pet Dataset, systematically searching for optimal hyperparameters, and finally, packaging the resulting model into a portable and user-friendly web application.

The fine-tuning is performed in 2 stages. First, the last layer (fully-connected) of the ResNet-34 is replaced by one with output dimension of 37, matching the number of classes of our dataset, with a dropaout layer preceding it. Then, this layer is trained on the dataset, leaving the rest frozen. Afterwards, the full model is trained on the dataset.

### Key Features
* **Hyperparameter Tuning:** A Bayesian sweep was conducted using Weights & Biases to find the optimal learning rates, dropout probability, and weight decay.
* **Interactive UI:** A simple and intuitive web interface built with Gradio allows users to upload an image and receive the top 3 breed predictions with confidence scores.
* **Containerized & Reproducible:** The entire application is containerized with Docker, ensuring that it runs consistently in any environment.
* **End-to-End Experiment Tracking:** All training runs, metrics, and configurations are logged in Weights & Biases for full transparency and reproducibility.

---

## 🛠️ Tech Stack & Architecture

The project is built with a modern stack for machine learning engineering:

* **Programming:** Python 3.11
* **ML Framework:** PyTorch & Torchvision
* **Experimentation:** Weights & Biases (W&B)
* **Web UI:** Gradio
* **Containerization:** Docker
* **Deployment:** Hugging Face Spaces & Git LFS

### High-Level Architecture
The workflow is designed to separate experimentation from deployment:

![Project Architecture Diagram](./assets/architecture.svg)

---

## ⚙️ Running the Project Locally

To set up and run this project on your local machine, follow these steps.

### 1. Prerequisites
* Conda (or another virtual environment manager)
* Docker Desktop installed and running
* A free Weights & Biases account

### 2. Clone the Repository
```bash
git clone https://github.com/nicdeluc/pet-breed-classifier.git
cd pet-breed-classifier
```

### 3. Set Up the Development Environment
This will install all the libraries needed for both training and inference.

```bash
# Create and activate the conda environment from the provided file
conda env create -f environment.yaml
conda activate pet-breed-clf

# Log in to your Weights & Biases account
wandb login
```

### 4. Run the Hyperparameter Sweep (Optional)
To replicate the experimentation phase, you can run a W&B sweep.

```bash
# 1. Initialize the sweep on the W&B server
# This will output a SWEEP_ID in the format: your-entity/your-project/sweep-id
wandb sweep src/sweep.yaml

# 2. Run the agent with the SWEEP_ID to start experiments
python src/sweep.py your-entity/your-project/sweep-id
```

### 5. Run the Application with Docker
The simplest way to run the final web application is with Docker. This uses the pre-trained model located in the assets/ folder.

```bash
# 1. Build the Docker image
docker build -t pet-classifier-app .

# 2. Run the Docker container
# The -p flag maps the container's port 7860 to your local port 7860
docker run -p 7860:7860 pet-classifier-app
```

You can now access the application by navigating to [http://localhost:7860] in your web browser.

## 📂 Repository Structure
The repository is organized to clearly separate concerns:

```text
.
├── assets/                 # Deployment assets (final model, example images)
├── notebooks/              # Jupyter notebooks for initial data exploration
├── src/                    # All source code
│   ├── app.py              # The Gradio application script
│   ├── sweep.py            # Script to run the W&B sweep agent
│   ├── train_final.py      # Script to train the final model with best params
│   ├── utils_app.py        # Helper functions for the Gradio app (inference)
│   ├── utils_train.py      # Helper functions for training and sweeps
│   └── sweep.yaml          # W&B sweep configuration file
├── .dockerignore           # Specifies files to exclude from the Docker image
├── .gitignore              # Specifies files to be ignored by Git
├── Dockerfile              # Instructions to build the application's Docker image
├── environment.yaml        # Conda environment for development
├── requirements_app.txt    # Minimal requirements for the Dockerized app
└── README.md               # This file
```

## 📈 Possible Improvements
Potential next steps could include:

- CI/CD Pipeline: Implement GitHub Actions to automatically build and test the Docker image on every push.

- Advanced Architectures: Experiment with more modern architectures like Vision Transformers (ViT) to compare performance.

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.
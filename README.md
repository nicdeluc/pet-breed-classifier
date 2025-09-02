# Pet Breed Classifier 🐕🐈

[![Status](https://img.shields.io/badge/Status-Completed-success)](https://github.com/your-username/your-repo-name)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

A complete MLOps project that fine-tunes a ResNet-34 model to classify 37 different breeds of cats and dogs. This repository covers the full lifecycle from experimentation and hyperparameter tuning with **Weights & Biases** to building and deploying a user-friendly web application with **Gradio** and **Docker**.

---
<!-- ## 🚀 Live Demo

You can try out the live application here: **[Hugging Face Spaces Demo Link]**


*A GIF or high-quality screenshot of your final Gradio application in action.*

--- -->
## 🛠️ Tech Stack

This project utilizes a modern stack for machine learning development and deployment:

* **Model Training & Experimentation:**
    * **PyTorch:** For building and fine-tuning the deep learning model.
    * **Weights & Biases (W&B):** For experiment tracking, hyperparameter sweeps, and model versioning.
    * **Jupyter Notebooks:** For initial data exploration and prototyping.

* **Application & Deployment:**
    * **Gradio:** To create a fast and intuitive web UI for the model.
    * **Docker:** To containerize the application for consistent, portable, and reproducible deployment.
    <!-- * **Hugging Face Spaces:** For hosting the final, public-facing application. -->

---
## ⚙️ Running the Project Locally

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

### 2. Set Up the Environment
It is highly recommended to use Conda for managing the environment.

```bash
# Create and activate the conda environment
conda env create -f environment.yaml
conda activate pet-breed-clf

# Log in to Weights & Biases
wandb login
```

### 3. Run the Hyperparameter Sweep (Optional)
To find the best hyperparameters, you can run the W&B sweep.

```bash
# Initialize the sweep (this will output a SWEEP_ID)
wandb sweep src/sweep.yaml

# Run the agent with the SWEEP_ID
wandb agent YOUR_SWEEP_ID
```
After the sweep, analyze the results in your W&B dashboard to find the optimal configuration.

### 4. Train the Final Model
Once you have the best hyperparameters, train the final model. You can either use the provided pre-trained model or run the `train_final.py` script after updating it with your best parameters.

```bash
python src/train_final.py
```
This will save the final model artifact (e.g., `best_model.pth`) to be used by the application.

### 5. Run the Gradio App with Docker
The easiest way to run the web application is using Docker.

```bash
# 1. Build the Docker image
docker build -t pet-classifier-app .

# 2. Run the Docker container
docker run -p 7860:7860 pet-classifier-app
```
You can now access the application by navigating to **http://localhost:7860** in your web browser.

---
## 📂 Project Structure

The repository is organized as follows:

```
.
├── data/                 # (Git-ignored) Where the dataset is stored
├── tuned_models/         # (Git-ignored) Where trained models are saved
├── notebooks/            # Jupyter notebooks for exploration and prototyping
├── src/                  # All source code
│   ├── app.py            # The Gradio application script
│   ├── sweep.py          # Script to run the W&B sweep agent
│   ├── fine_tune.py      # Script to fine tune a model with specific hyperparameters
│   ├── utils.py          # Helper functions for data, training, etc.
│   └── sweep.yaml        # W&B sweep configuration file
├── .dockerignore         # Specifies files to exclude from the Docker image
├── .gitignore            # Specifies files to be ignored by Git
├── Dockerfile            # Instructions to build the application's Docker image
├── environment.yaml      # Conda environment specification for development
├── requirements_app.txt  # Minimal requirements for the Dockerized app
└── README.md             # This file
```

---
## 📄 License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
# AI/ML Project Template Repository

## Overview

Welcome to the LLM serving repository! This repository is ispired by the course [Efficiently Serving LLMs](https://learn.deeplearning.ai/courses/efficiently-serving-llms) from DeepLearning.AI for an efficient LLM serving using transformers and pytorch.

## Features

- **various implimentations**: The code for several different versions of LLM serving is implimented so they can be chosen and used based on prefrences and capacity.
- **Well documented&**: The code base is well commented and documented with tests provided.
- **KV cashing**: A code based understading of KV cashing and how it can be helpful with efficiency is provided
- **Batching**:  

## Getting Started

To get started with this project using the repository, follow these steps:

1. **Clone the Repository**: Clone the created repository to your local machine using Git.

   ```bash
   git clone https://github.com/rashidis/LLM-serving.git
2. **Navigate to the Project Directory**: Enter the project directory in your terminal or command prompt.
3. **Install Dependencies**: Create the conda environment with dependencies installed:

   ```bash
   conda env create -f environment.yml
4. **Activate the conda environment**:

   ```bash
   conda activate income-prediction-env
5. **Run code**: cd to the src folder and use python to run code

   ```bash
   cd src
   python *.py
6. **Contribute**: If you've made improvements or additions to the template, consider contributing back to the community by submitting a pull request.

## Directory Structure

The repository is organized as follows:

- **`data/`:** 
- **`models/`:** Directory for storing a local version of the trained models.
- **`notebooks/`:** 
- **`src/`:** Python scripts for modularized code, including data preprocessing, feature engineering, and model training.</br>
   |__ `gpt2_chat_completion.py` : Using gpt2 and kv cashing for efficient next token generation and chat completion. </br>
   |__ `gpt2_chat_batching.py` : Using gpt2 and batching for high throughput and low latency next token generation and chat completion.

- **`tests/`:**  
- **`results/`:** 
- **`README.md`:** Project overview and usage
- **`LICENSE`:** License file
## Contribution Guidelines

We welcome contributions from the community to improve this template repository. If you have suggestions, bug fixes, or additional features to add, please follow these guidelines:

- Fork the repository and create a new branch for your contribution.
- Make your changes, ensuring they adhere to the project's coding style and conventions.
- Test your changes thoroughly.
- Update documentation if necessary.
- Submit a pull request, providing a detailed description of your changes.

## License

This project is licensed under the [MIT License](License). Feel free to use, modify, and distribute this template for your AI/ML projects.
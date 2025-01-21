# **Privacy-Preserving Fraud Detection in Banking Using Horizontal Federated Learning (HFL)**

This project implements **Horizontal Federated Learning (HFL)** for privacy-preserving fraud detection in the banking and fintech domain. By training machine learning models collaboratively across institutions without sharing raw data, it addresses critical issues such as privacy, regulatory compliance, and data heterogeneity.

---

## **Features**
- Privacy-preserving fraud detection using Horizontal Federated Learning.
- Support for simulation of multiple clients in federated learning.
- Designed to work seamlessly within Jupyter Notebook for interactive experimentation and visualization.

---

## **Getting Started**

Follow these steps to set up and run the project.

### **1. Prerequisites**
Ensure you have the following installed:

- `uv` for virtual environment management:
  ```bash
  pip install uv
  ```

---

### **2. Setup Instructions**

#### **Step 1: Initialize the Project Environment**
1. Open your terminal and create a project folder:
   ```bash
   uv init project_name
   cd project_name
   ```

2. Clone the GitHub repository into your project directory:
   ```bash
   git clone https://github.com/Federated-Learning-MLC/loan-fintech-hfl.git
   ```

---

#### **Step 2: Install Dependencies**
Use `uv` to install all project dependencies:
```bash
uv install
```

---

#### **Step 3: Run the Jupyter Notebook**
1. Start the Jupyter Notebook server:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `notebooks/` folder and open the relevant notebook (e.g., `01_EDA.ipynb`).
3. Run all the cells sequentially:
   - The initial cells will install dependencies and set up the `data/` directory.
   - The required folder structure will be automatically created.

---

#### **Step 4: Add the Dataset**
1. Download the **Base.csv** dataset from Kaggle:
   - Dataset link: [Bank Account Fraud Dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data?select=Base.csv)

2. Place the `Base.csv` file into the `data/` directory created in the previous step.

---

#### **Step 5: Continue Running the Notebook**
After adding the dataset:
1. Restart the Jupyter Notebook kernel.
2. Re-run all cells to preprocess the data and simulate federated learning clients.
3. Run the training and evaluation sections to complete the federated learning workflow.

---

### **Project Structure**

The repository is organized as follows:
```
loan-fintech-hfl/
│
├── data/                # Directory containing the dataset (Base.csv)
├── notebooks/           # Jupyter notebooks for EDA and experimentation
├── src/                 # Source code for the project
│   ├── __init__.py
│   ├── config.py        # Project configurations
│   ├── data_prep.py     # Script to prepare data for FL
│   ├── local_utility.py # Utility functions
│   ├── paths.py         # Script to set up folder structure
│
├── .gitignore           # Ignore rules for Git
├── README.md            # Project documentation
├── pyproject.toml       # Dependency and environment setup
├── uv.lock              # Environment lock file
└── LICENSE              # License for the project
```

---

## **How to Contribute**
We welcome contributions to improve this project! Follow these steps to contribute:
1. Fork the repository on GitHub.
2. Clone your fork to your local machine:
   ```bash
   git clone https://github.com/your-username/loan-fintech-hfl.git
   ```
3. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature-name
   ```
4. Commit your changes:
   ```bash
   git commit -m "Add a new feature"
   ```
5. Push the changes to your fork:
   ```bash
   git push origin feature-name
   ```
6. Open a pull request on the original repository.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## **Acknowledgments**
- The dataset used in this project is from the [Bank Account Fraud Dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022).

---

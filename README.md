<div align="center">
<h1 align="center">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
<br>NLP_PHISHING_CLASSIFICATION</h1>
<h3>◦ Developed with the software and tools below.</h3>

<p align="center">
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat-square&logo=Jupyter&logoColor=white" alt="Jupyter" />
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white" alt="pandas" />
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat-square&logo=NumPy&logoColor=white" alt="NumPy" />
</p>
</div>


---

## 📖 Table of Contents
- [📖 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [📧 Spam Classification](#-spam-classification)
- [📦 Features](#-features)
- [📂 repository Structure](#-repository-structure)
- [⚙️ Modules](#️-modules)
- [🔧 Installation](#-installation)
- [🛣 Roadmap](#-roadmap)
- [📈 Model Performance](#-model-performance)
- [🤝 General Conclusion](#-contributing)
- [👏 Acknowledgments](#-acknowledgments)

---


## 📍 Overview

Basic NLP project to classify phishing emails from spam emails. This project is based on the [Spam Email raw text for NLP](https://www.kaggle.com/datasets/subhajournal/phishingemails) dataset from Kaggle. The dataset contains 18,650 emails, 7,000 of which are spam emails and others are not spam emails

This project is my report for my NLP course at the EPF school during my 5th year of engineering school. 

It is a first foot in the NLP world, so I tried to use the most basic models and techniques to classify the emails with ML models. I also tried Deep Learning for the first time.

---
## 📧 Spam Classification

This Project is based on Spam Email detection. The goal is to classify emails as spam or not spam. To do that we will use different NLP techniques and models.

Here is an explanation on how to classify emails as spam or not spam:

<img src="https://raw.githubusercontent.com/deepankarkotnala/Email-Spam-Ham-Classifier-NLP/master/images/email_spam_ham.png"/>

---


## 📦 Features

- `Data Exploratory Analysis`
- `Basic ML model`
- `Improving basic ML performance by hyperparameter tuning, learning curve ...`
- `Basic Deep Learning model`

---


## 📂 Repository Structure

```
└── NLP_Phishing_Classification/
    ├── data/
      ├── Phishing_Email.csv
      └── Spam Email raw text for NLP.csv
    ├── images/
    ├── models/
    ├── packages/
    ├── 1_Exploratory_data_analysis.ipynb
    ├── 2_basic_model.ipynb
    ├── 3_deep_learning.ipynb
    ├── requirements.txt
    └── utils.py

```

---


## ⚙️ Modules

| File                                                                                                                                       | Summary                   |
| ---                                                                                                                                        | ---                       |
| [1_Exploratory_data_analysis.ipynb](https://github.com/maxpline83/NLP_Phishing_Classification/blob/main/1_Exploratory_data_analysis.ipynb) | Exploration of the dataset |
| [2_basic_model.ipynb](https://github.com/maxpline83/NLP_Phishing_Classification/blob/main/2_basic_model.ipynb)                             | Basic ML model & Hyperparameter tuning|
| [3_deep_learning.ipynb](https://github.com/maxpline83/NLP_Phishing_Classification/blob/main/3_deep_learning.ipynb)                         | Basic Deep Learning model|
| [requirements.txt](https://github.com/maxpline83/NLP_Phishing_Classification/blob/main/requirements.txt)                                   | Python libraries used  |
| [utils.py](https://github.com/maxpline83/NLP_Phishing_Classification/blob/main/utils.py)                                                   | Usefull python script for this project |

---

### 🔧 Installation

1. Clone the NLP_Phishing_Classification repository:
```sh
git clone https://github.com/maxpline83/NLP_Phishing_Classification
```

2. Change to the project directory:
```sh
cd NLP_Phishing_Classification
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```



---


## 🛣 Project Roadmap

> - [X] `ℹ️ Task 1: Comment the code`
> - [ ] `ℹ️ Task 2: change basic script to function & Class`
> - [ ] `ℹ️ Task 3: Add a new feature in Basic Model`
> - [X] `ℹ️ Task 4: Improve the Deep Learning model`


---

## 📈 Model Performance

| Model                        | Accuracy (%) | Macro F1-score (%) | Training size | Training time per epochs (s) |
|------------------------------|--------------|--------------------|---------------------------|-----------------------------|
| Basic Model                  | 94           | 95                 | 1000                      | 1,2                      |
| Simple Neural Network        | 77           | 80                 | 1000                      | 1                      |
| RNN                          | 80           | 83                 | 1000                      | 3                  |
| LSTM                         | 83           | 86                 | 1000                      | 4                  |
| CNN                          | 81           | 84                 | 1000                      | 1                  |

---


## 🤝 General Conclusion

This project presents an initial foray into the world of NLP, employing fundamental NLP techniques and models to classify emails as spam or not. It provides insights into the performance of basic models and serves as a stepping stone for future improvements

The basic model is a good start to classify emails as spam or not. Also It is a good model to start with and to have a first idea of the performance of the model but I think that Deep Leraning models are more efficient for this kind of problem. I just don't have enough time to improve the Deep Learning model now.

But the result that I have are not really relevant, Simple Neural Network is perfomant but has a larger training time. I think that I can improve the model by changing the Deep Learning model and the NLP encoding methods. I will try to improve the model in the future.

With testing 2 others type of Deep Learning model, I can see that the LSTM model is more efficient than the RNN model. How ever, the performance are not really good compared to basic model. It may be due to the NLP encoding methods or the size of the trainning dataset. We know that Deep Learning model need a lot of data to be efficient.

---

## 👏 Acknowledgments

- [Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails) for the dataset
- [ChatGPT](https://chat.openai.com/) to get some help with the Deep Learning model and the different NLP encoding methods
- [Medium](https://medium.com/analytics-vidhya/nlp-text-encoding-a-beginners-guide-fa332d715854) for the NLP encoding methods
- [Towards Data Science](https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa) for the Deep Learning model
- [NLP courses from Ryan Pegoud](https://github.com/RPegoud) for the NLP course and the different NLP methods (preprocessing python script, etc.)

[**Return**](#Top)

---


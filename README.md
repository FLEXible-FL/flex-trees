# flex-trees

The flex-trees package consists of a set of tools and utilities to work with Decision Tree (DT) models in Federated Learning (FL). It is designed to be used with the [FLEXible](https://github.com/FLEXible-FL/FLEXible/) framework, as it is an extension of it.

flex-trees comes with some state-of-the-art decision tree models for federated learning. It also provides multiple tabular datasets to test the models.

The methods implemented in the repository are:
| `Model`            | `Description`      | `Citation`              |
| ------------------ | :------------------: | -------------------: |
| Federated ID3 | The ID3 model adapted to a federated learning scenario. | [A Hybrid Approach to Privacy-Preserving Federated Learning](https://arxiv.org/pdf/1812.03224.pdf) |
| Federated Random Forest | The Random Forest (RF) model adapted to a federated learning scenario. Each client builds a RF locally, then `N` trees are randomly sampled from each client to get a global RF composed from the `N` trees retrieved from the clients. | [Federated Random Forests can improve local performance of predictive models for various healthcare applications](https://pubmed.ncbi.nlm.nih.gov/35139148/) |
| Federated Gradient Boosting Decision Trees | The Gradient Boosting Decision Trees model adapted to a federated learning scenario. In this model a global hash table is first created to aling the data between the clients within sharing it. After that, `N` trees (CART) are built by the clients. The process of building the ensemble is iterative, and one client builds the tree, then it is added to the ensemble, and after that the weights of the instances is updated, so the next client can build the next tree with the weights updated.| [Practical Federated Gradient Boosting Decision Trees](https://arxiv.org/abs/1911.04206) |

The tabular datasets available in the repository are:
| `Dataset`            | `Description`      | `Citation`              |
| ------------------ | :------------------: | -------------------: |
| Adult | The Adult dataset is a dataset that contains demographic information about the people, and the task is to predict if the income of the person is greater than 50K. | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult) |
| Breast Cancer | The Breast Cancer dataset is a dataset that contains information about the breast cancer, and the task is to predict if the cancer is benign or malignant. | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) |
| Credit Card | The Credit Card dataset is a dataset that contains information about the credit card transactions, and the task is to predict if the transaction is fraudulent or not. | [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) |
| ILPD | The ILPD dataset is a dataset that contains information about the Indian Liver Patient, and the task is to predict if the patient has liver disease or not. | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/ILPD+%28Indian+Liver+Patient+Dataset%29) |
| Nursery | The Nursery dataset is a dataset that contains information about the nursery, and the task is to predict the acceptability of the nursery. | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/nursery) |
| Bank Marketing | The Bank Marketing dataset is a dataset that contains information about the bank marketing, and the task is to predict if the client will subscribe to a term deposit. | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) |
| Magic Gamma | The Magic Gamma dataset is a dataset that contains information about the magic gamma, and the task is to predict if the gamma is signal or background. | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope) |

##  Tutorials

To get started with flex-trees, you can check the [notebooks](https://github.com/FLEXible-FL/flex-trees/tree/main/notebooks) available in the repository. They cover the following topics:

- [Federated ID3 with FLEXible](https://github.com/FLEXible-FL/flex-trees/blob/main/notebooks/Federated%20Random%20Forest%20with%20FLEX.ipynb).
- [Federated Random Forest with FLEXible](https://github.com/FLEXible-FL/flex-nlp/blob/main/notebooks/Federated%20QA%20with%20Hugginface%20using%20FLEXIBLE.ipynb).
- [Practical Federated Gradient Boosting Decision Trees with FLEXible](https://github.com/FLEXible-FL/flex-trees/blob/main/notebooks/Federated%20Gradient%20Boosting%20Decision%20Trees%20with%20FLEX.ipynb).

## Installation

We recommend Anaconda/Miniconda as the package manager. The following is the corresponding `flex-trees` versions and supported Python versions.

| `flex`            | `flex-trees`      | Python              |
| :------------------: | :------------------: | :-------------------: |
| `main` / `nightly` | `main` / `nightly` | `>=3.8`, `<=3.11`   |
| `v0.6.0`           | `v0.1.0`           | `>=3.8`, `<=3.11`    |

To install the package, you can use the following commands:

Using pip:
```
pip install flextrees
```

Download the repository and install it locally:
```
git clone git@github.com:FLEXible-FL/flex-trees.git
cd flex-trees
pip install -e .
```


## Citation

If you use this package, please cite the following paper:

``` TODO: Add citation ```
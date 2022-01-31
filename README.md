# Arsenal

**This is a collection of many useful machine learning models**

**Thus, it is named as Arsenal!**

**The content of this collection is shown as below:**

## Regression

> + OLS——Ordinary Least Square
>
> + RR——Ridge Regression
>
> + LASSO——Least Absolute Shrinkage and Selection Operator
> + PLSR——Partial Least Square Regression
> + GPR——Gaussian Process Regression
> + ELM——Extreme Learning Machine
> + MC-GCN——Multi-Channel Graph Convolutional Networks
> + GC-LSTM——Graph Convolution Long Short-Term Memory

## Classification

> + LR——Logistic Regression

## Regression & Classification

> + FCN——Fully Connected Networks
> + LSTM——Long Short-Term Memory
> + GCN——Graph Convolutional Networks

## Dimensionality Reduction

> + PCA——Principal Component Analysis
> + t-SNE——t-distributed Stochastic Neighbor Embedding
> + AE——Auto-Encoders
> + VAE——Variational Auto-Encoders

**To be continued ...**

## Functions

**Arsenal can solve regression, classification or dimensionality reduction problem:**

```powershell
# Regression
python main.py -model OLS -prob regression

# Classification
python main.py -model LR -prob classification

# Dimensionality reduction
python main.py -model PCA -prob dimensionality-reduction
```

**Also, some models are implemented by myself in Arsenal, which can be easily used**

```powershell
# Default is using the official package
python main.py -model PLSR -prob regression

# Models implemented by myself can be used via "-myself" argument
python main.py -model PLSR -prob regression -myself True
```

**What's more, the hyper-parameters in models can also be optimized through grid search or random search**

```powershell
# Grid search
python main.py -model LASSO -prob regression -hpo True -hpo_method GS

# Random search
python main.py -model LASSO -prob regression -hpo True -hpo_method RS
```

**Last but not least, in some regression problems with multiple labels, we proposed two models, MC-GCN and GC-LSTM, to consider the correlations among multiple labels for better performance**

```powershell
# MC-GCN: the correlations among multiple labels are considered in modelling
python main.py -model MCGCN -prob regression -multi_y True

# GC-LSTM: the correlations among multiple labels and the temporalities among samples are both considered in modelling
python main.py -model GCLSTM -prob regression -multi_y True
```

**Thanks for your attention and support!** 
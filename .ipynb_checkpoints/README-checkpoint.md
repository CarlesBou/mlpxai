# Achieving faithful explainability in feedforward neural networks through accurately computed feature attribution
This repository contains the supplementary material for the paper (currently under review):

Carles-Bou, Jose L. and Carmona, Enrique J. **Achieving faithful explainability in feedforward neural networks through accurately computed feature attribution**. 

## Paper Abstract
The rapid advancements in machine learning have led to the deployment of complex models in critical domains such as healthcare, finance, and autonomous systems. Despite their remarkable predictive performance, the opaque nature of these models presents significant challenges for interpretability, which is essential for trust, accountability, and regulatory compliance. Explainable Artificial Intelligence (XAI) has emerged as a crucial field addressing these challenges by making black-box models more transparent. In this paper, we propose a novel model-specific local post-hoc explanation method for feedforward neural networks (FNNs) built on a solid mathematical foundation. Our approach enables the exact computation of input feature attributions for individual predictions and achieves perfect fidelity in replicating model behavior. These two properties, combined with competitive computational efficiency, demonstrate the superior performance of the proposed method compared to state-of-the-art XAI techniques. We validate the method through extensive experiments, showing its versatility across diverse types of problems. This work enhances interpretability and trust in AI systems by providing a reliable explanation framework applicable across a wide range of scenarios modeled with FNNs.


## Repository description
The repository contains the source files developed to implement the new local post-hoc explainer for Feed-forward Neural 
Networks (FNNs) named "**Feature Attribution Computed Exactly (FACE)**" and the code of different examples showing the 
use of the explainer over regression and classification tabular problems and in classifation with images.

It has been implemented in Python and uses Keras and Tensorflow framework. We are working in a new version compatible
with PyTorch, too.

The */source/mlpxai/explainers/face* folder keeps the source files of the new explainer.

The */source/examples* folder contains Python programs that show how to use FACE.

And, in the */source/examples/notebooks* folder, you can find the same Python programs but in Jupyter Notebook format.



## 


## Licenses
This work is licensed under Creative Commons Zero v1.0 Universal (or any later version) license.
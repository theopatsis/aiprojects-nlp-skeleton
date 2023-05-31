## Overview
ML model that classifies sincere and insincere Quora entries using GloVe embeddings and Pytorch RNNs. Achieved a 95% accuracy despite our skewed dataset of 1.3 million entries by examining the confusion matrix and using a clever way of upsampling the data for insincere entries. The project was made as part of  the ACM AI Projects Track in Winter 2023.

## Installation
First, clone the GitHub repo and move to the correct directory.
```bash
  git clone https://github.com/theopatsis/aiprojects-nlp-skeleton.git
  cd aiprojects-nlp-skeleton
  conda activate env
```

Next, install all required libraries and run it.
```bash
  conda install pytorch torchvision pandas matplotlib jupyter tqdm tensorboard transformers torchmetrics
  python main.py
```
The code will run through all batches and print training accuracy, F1 score and the confusion matrix for each batch.

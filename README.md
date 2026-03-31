# DiscourseQualityIndex: Multi-Output BERT Regression

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![Weights & Biases](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/)

##   Project Overview

The **DiscourseQualityIndex** is an advanced Natural Language Processing (NLP) pipeline that evaluates human discourse (e.g., Reddit comments, BlueSky posts) across **6 distinct dimensions simultaneously**.

Instead of training 6 separate models—which is computationally expensive and ignores shared linguistic traits—this project implements a **"Shared Brain" Multi-Task Learning architecture**. By leveraging a single `bert-base-uncased` body with a 6-node regression head, the model learns the fundamental rules of human language once and applies those contextual weights to all 6 scoring tasks in parallel.

###   The 6 Dimensions of Discourse Quality

| Dimension | Description |
|-----------|-------------|
| **Level of Justification** | How well the argument is supported with evidence |
| **Respect Towards Demands** | Civility when addressing requests or requirements |
| **Respect Towards Counterarguments** | Acknowledgment and respectful treatment of opposing views |
| **Content of Justification** | Quality and relevance of reasoning provided |
| **Respect Towards Groups** | Civility when discussing social or demographic groups |
| **Constructive Politics** | Contribution to productive political discourse |

##   Key Engineering Features

- **High-Speed Data Vectorization**: Utilizes Pandas vectorization to "zip" 6 continuous target variables into 1D PyTorch tensors for seamless Hugging Face tokenization across 200,000+ rows

- **Custom PyTorch Interceptor (PureMSETrainer)**: Overrides default Hugging Face loss calculations. Extracts and logs individual Training Mean Squared Error (MSE) for all 6 dimensions before PyTorch averages them into a master loss

- **Granular Validation Metrics**: Implements a custom Scikit-Learn `compute_metrics` grader to calculate independent Validation MSE for every dimension to strictly monitor for overfitting

- **Multi-GPU Ready**: Configured for Data Parallelism on enterprise hardware (e.g., 4x NVIDIA L40S), optimizing batch sizes (256 effective) to prevent CPU/GPU bottlenecks

- **Automated Artifact Smoothing**: Includes a custom Matplotlib pipeline to dynamically clean "Frozen Time" logging artifacts (stacking glitches during evaluation steps), generating clean, presentation-ready learning curves

##   Installation & Setup

### Clone the repository:

```bash
git clone https://github.com/niyimoluga/DiscourseQualityIndex.git
cd DiscourseQualityIndex

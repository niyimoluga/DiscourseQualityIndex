DiscourseQualityIndex: Multi-Output BERT Regression
Project Overview
The DiscourseQualityIndex is an advanced Natural Language Processing (NLP) pipeline that evaluates human discourse (e.g., Reddit comments, BlueSky posts) across 6 distinct dimensions simultaneously.

Instead of training 6 separate models—which is computationally expensive and ignores shared linguistic traits—this project implements a "Shared Brain" Multi-Task Learning architecture. By leveraging a single bert-base-uncased body with a 6-node regression head, the model learns the fundamental rules of human language once and applies those contextual weights to all 6 scoring tasks in parallel.

The 6 Dimensions of Discourse Quality


Level of Justification

Respect Towards Demands

Respect Towards Counterarguments

Content of Justification

Respect Towards Groups

Constructive Politics

 Key Engineering Features
High-Speed Data Vectorization: Utilizes Pandas vectorization to "zip" 6 continuous target variables into 1D PyTorch tensors for seamless Hugging Face tokenization across 200,000+ rows.

Custom PyTorch Interceptor (PureMSETrainer): Overrides default Hugging Face loss calculations. Extracts and logs individual Training Mean Squared Error (MSE) for all 6 dimensions before PyTorch averages them into a master loss.

Granular Validation Metrics: Implements a custom Scikit-Learn compute_metrics grader to calculate independent Validation MSE for every dimension to strictly monitor for overfitting.

Multi-GPU Ready: Configured for Data Parallelism on enterprise hardware (e.g., 4x NVIDIA L40S), optimizing batch sizes (256 effective) to prevent CPU/GPU bottlenecks.

Automated Artifact Smoothing: Includes a custom Matplotlib pipeline to dynamically clean "Frozen Time" logging artifacts (stacking glitches during evaluation steps), generating clean, presentation-ready learning curves.

🛠️ Installation & Setup
Clone the repository:

Bash
git clone https://github.com/niyimoluga/DiscourseQualityIndex.git
cd DiscourseQualityIndex
Install dependencies:

Bash
pip install datasets pandas transformers scikit-learn scipy matplotlib wandb torch
Weights & Biases Setup:
Ensure you have a W&B account for live tracking. The script expects an API key:


 Pipeline Architecture
The training script (train.py) is structured into 6 sequential phases:

Data Acquisition: Pulls 200k labeled rows from the Hugging Face hub (lanretto/discourse_quality).

Transformation: Bundles the 6 target columns into a single label tensor and performs an 80/20 train/test split.

Tokenization: Formats text via the bert-base-uncased tokenizer and locks the outputs as exact PyTorch tensors.

Model Initialization: Loads the AutoModel configured for problem_type="regression" to force MSE loss across all 6 outputs.

Custom Grader & Interceptor: Injects the PureMSETrainer and compute_metrics logic to intercept dimension-specific losses.

Training & Plotting: Executes the training loop, syncs to W&B, and renders local Matplotlib charts for both Master Loss and Per-Dimension Loss.

Results & Visualization
The custom plotting scripts at the end of the pipeline automatically generate a 2x3 grid tracking the Training Loss vs. Validation Loss for every individual dimension.

By tracking these independently, the pipeline guarantees Zero Overfitting—proving the model learns generalized linguistic rules rather than simply memorizing the training dataset.

Developed for research into algorithmic content recommendation and digital discourse.

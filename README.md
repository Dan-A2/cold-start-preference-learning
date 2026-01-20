# Cold Start Active Preference Learning in Socio-Economic Domains

This repository contains the official source code and experimental setup for the paper:  
**"Cold Start Active Preference Learning in Socio-Economic Domains"**, submitted to the ....

![Framework Overview](Images/framework_diagram.png)

---

## Abstract

Active preference learning offers an efficient approach to modeling preferences, but it is hindered by the cold-start problem, which leads to a marked decline in performance when no initial labeled data are available. While cold-start solutions have been proposed for domains such as vision and text, the cold-start problem in active preference learning remains largely unexplored, underscoring the need for practical, effective methods. Drawing inspiration from established practices in social and economic research, the proposed method initiates learning with a selfsupervised phase that employs Principal Component Analysis (PCA) to generate initial pseudo-labels. This process produces a “warmed-up” model based solely on the data’s intrinsic structure, without requiring expert input. The model is then refined through an active learning loop that strategically queries a simulated noisy oracle for labels. Experiments conducted on various socioeconomic datasets, including those related to financial credibility, career success rate, and socio-economic status, consistently show that the PCA-driven approach outperforms standard active learning strategies that start without prior information. This work thus provides a computationally efficient and straightforward solution that effectively addresses the cold-start problem.

---

## Framework Overview

Our proposed framework consists of four main stages designed to efficiently learn preferences from a cold start:

1. **Data Preparation**  
   Raw data is cleaned, preprocessed, and standardized. Categorical features are intelligently encoded into numerical or one-hot representations.

2. **Warm-Start Pre-training**  
   A self-supervised phase where Principal Component Analysis (PCA) is used to generate pseudo-labels. An initial XGBoost model is pre-trained on these labels to give it a "warm start."

3. **Simulated Expert Oracle**  
   An oracle that mimics a real-world expert by providing preference labels with stochastic noise, modeled using the Bradley-Terry model.

4. **Training Loop**  
   The warm-started model is incrementally refined by strategically querying the oracle for new labels, focusing on the most informative data pairs.

---

## Repository Structure

```
.
├── Config/
│ └── util.py           # Utility functions
├── Datasets/           # Directory for datasets
├── FIFA/               # Initial Jupyter notebooks for the FIFA dataset (Different experiments were done here)
├── Images/             # Output directory for generated plot images
├── Plots/              # Output directory for plots' data
├── *.ipynb             # Notebooks for datasets
├── dopewolf.py.        # Code to run dopewolf method
├── plot_generator.py   # Code to generate plots from saved data
├── README.md           # This file
└── run.py              # Standalone script to execute the framework on a cleaned dataset
```

---

## Download Datasets

Please download the datasets used in the study and place them in the `Datasets/` directory.

Datasets can be downloaded directly **except** the Household and FIFA 22 datasets due to their large size. You can download them from the following links:

**[Download Household Dataset](https://huggingface.co/datasets/Dan-A2/Iranian-Household-Data)**<br />
**[Download FIFA 22 Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset)**

Here are the links to other datasets' main pages:<br />
**[Download Credit Dataset](https://www.kaggle.com/datasets/uciml/german-credit)**<br />
**[Download Happiness Dataset](https://www.kaggle.com/datasets/mathurinache/world-happiness-report-20152021)**<br />
**[Download Student Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset)**

---

## Usage and Replication

The experiments can be run from the notebooks. The core logic for running a comparative experiment is outlined in the `run.py` script.

### Replicating Paper Results

To replicate the figures from the paper, run the script for all policies across the relevant datasets. The results and plots will be saved to the `plots/` directory.

---

## Citation

If you use this code or our framework in your research, please cite our paper:

```
@misc{fayazbakhsh2025coldstartactivepreference,
      title={Cold Start Active Preference Learning in Socio-Economic Domains}, 
      author={Mojtaba Fayaz-Bakhsh and Danial Ataee and MohammadAmin Fazli},
      year={2025},
      eprint={2508.05090},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.05090}, 
} 
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
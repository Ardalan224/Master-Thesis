# Optimal Dataset Size for Recommender Systems: Evaluating Algorithms' Performance via Downsampling
This public GitHub repository contains the Python code used for the experiments conducted in my Master's thesis, titled: "Optimal Dataset Size for Recommender Systems: Evaluating Algorithms' Performance via Downsampling."

The programs here provide a general structure of the code used in the experiments. Adjustments may have been made in specific scenarios or experimental requirements. Note that the exact set of random seeds and hyperparameters used in the experiments may differ from those specified in the uploaded scripts.

Feel free to explore the repository and the structure of the experiments conducted. Below are additional clarifications and details:

Key Details and Clarifications
1- Random Seeds

  - All experiments in the thesis were conducted using 5 random seeds: 21, 42, 63, 84, and 105.
  - The Python scripts in this repository reflect only one of these seeds as an example. To reproduce the full experiment, you can modify the seed value in the     
    code.
2- Downsampling Portions and Split Setup

Each downsampling portion results in a specific split setup: 10% Test Set, 10% Validation Set, and a varying portion of the Training Set.
In the code, specific sequences of values and their corresponding proportions (e.g., # 28 ---> 0.56, 36 ---> 0.385) are carefully designed to achieve the desired final split for each downsampling portion. When all cells are executed with the specified values for a given portion, the resulting allocation of interactions across the training, validation, and test sets aligns with the intended setup.
3- Core Pruning Levels

Experiments were conducted with both 10-core and 30-core pruning levels.
The scripts in this repository default to the 10-core pruning value. You can adjust this to 30 in the script to replicate experiments with 30-core pruning.
4- Hyperparameter Tuning

The hyperparameter values and configurations specified in the scripts represent the general range of parameters explored during tuning.
The final hyperparameter values used in the experiments might differ slightly based on the algorithm, dataset, or scenario.
5- Figures Directory

The folder named "Figures" contains comprehensive plots generated from the experiments. These include:
Detailed individual behaviors of each algorithm on each dataset across all downsampling portions and pruning levels.
Many plots not included in the final thesis due to the extensive number of results.
This directory provides a deeper exploration of algorithm performance for those interested in further analysis.

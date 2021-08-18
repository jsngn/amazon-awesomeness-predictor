# Amazon Awesomeness Predictor

## Description
This repository contains the code for data exploration & preprocessing, parameter tuning, and different approaches attempted to predict whether a product on Amazon is awesome or not (binary classification). Awesomeness is defined as whether the product's average review > 4.5 stars (in the available dataset). The final model (`final_iteration_model.py`) predicts using unseen training data (from `full_train.csv`) with an average weighted F-1 score of 0.76 (average result from 10-fold cross validation). On test data (from `Test.csv`) it predicts with a weighted F-1 score of 0.73.

The code in this repository is for a group project & personal extra credit submission for COSC 74 at Dartmouth College.

## Instructions

1. Clone the repo.

2. `cd amazon-awesomeness-predictor`.

3. `tar -xf data/Test.tar.gz` and `tar -xf data/Test.tar.gz`; this should extract `full_train.csv` and `Test.csv` into the same directory as `main.py`.

4. Run `main.py` with `full_train.csv` and `Test.csv` in the same directory to see the final model's performance (precision, recall, F-1) in each of the 10 folds. Change the third param in `main.py` from `xgb10` to `predict` then rerun to produce the predictions CSV file for test data in `Test.csv` using the final model. The output file is formatted as follows: 1st column - automatic indexing by Pandas; 2nd column - Amazon product ID; 3rd column - Awesomeness.

If you would like to see the performance of the earlier models (`first_iteration_model.py` or `second_iteration_model.py`), simply run those files with `full_train.csv` in the same directory. These models were not adapted to produce a predictions CSV for `Test.csv` data because they were used not in the end.

## Data Files

`full_train.csv`: training data for the models, contains data about Amazon product reviews e.g. review text, review summary, product price, etc. The binary `target` column indicates whether the product is awesome across all its reviews.

`Test.csv`: unlabeled test data.

Note that these have been compressed (separately due to upload size limits for individual files). The compressed files are in `data/`. Please follow the instructions above to extract them.

## Code Files

`final_iteration_model.py`: code of the final model (contains both group project & personal extra credit code).

`second_iteration_model.py`: code of my group's second approach.

`first_iteration_model.py`: code of my group's first approach.

`common.py`: common code used by the different iterations above.

`data_exploration.py`: code of my group's data exploration, with comments explaining observations and conclusions.

`data_preprocessing.py`: preprocessing code, with comments explaining observations and conclusions. Most of this code is unused in the final model; most preprocessing done in the final model is directly included (with documentation) in `final_iteration_model.py`.

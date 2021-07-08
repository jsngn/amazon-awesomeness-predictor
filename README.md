# Amazon Awesomeness Predictor
Repo currently being updated.

## Description
This repository contains the code for data exploration & preprocessing, parameter tuning, and different approaches to predict whether a product on Amazon is awesome or not (binary classification), defined as whether its average (in the available dataset) review > 4.5 stars. The final model (`final_iteration_model.py`) predicts on unseen training data (from `full_train.csv`) with an average weighted F-1 score of 0.76 (average result from 10-fold cross validation). On test data (from `Test.csv`) it predicts with a weighted F-1 score of 0.73.

This code is my group project & extra credit submission for COSC 74 at Dartmouth College.

## Instructions

Run `main.py` with `full_train.csv` and `Test.csv` in the same directory to see the final model's performance (precision, recall, F-1) in each of the 10 folds. Change the third param in `main.py` from `xgb10` to `predict` rerun to produce the predictions CSV file for test data in `Test.csv` using the final model. The output file is formatted as follows: 1st column - automatic indexing by Pandas; 2nd column - Amazon product ID; 3rd column - Awesomeness.

If you would like to see the performance of the earlier models (`first_iteration_model.py` or `second_iteration_model.py`), simply run those files with `full_train.csv` in the same directory. These models were not adapted to produce a predictions CSV for `Test.csv` data because they were used not in the end.

## Data Files

`full_train.csv`: training data for the models, contains data about Amazon product reviews e.g. review text, review summary, product price, etc. The binary `target` column indicates whether the product is awesome across all its reviews.

`Test.csv`: unlabeled test data.

## Other Files

`final_iteration_model.py`: code of the final model (group project & extra credit code).

`second_iteration_model.py`: code of my group's second approach.

`first_iteration_model.py`: code of my group's first approach.

`common.py`: common code used by different iterations.

`data_exploration.py`: code of my group's data exploration, with comments explaining observations and conclusions.

`data_preprocessing.py`: preprocessing code, with comments explaining observations and conclusions. Most of this code is unused in the final model; most preprocessing done in the final model is directly included (with documentation) in `final_iteration_model.py`.

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import statistics
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from data_preprocessing import DataPreprocessing
from common import Common
import xgboost as xgb


class FinalIterationModel:
    """
    Final model with extra credit & previous group project code. Model predicts using (preprocessed) reviewText,
    summary, review count, root-genre, and related. This file contains code for model selection, hyperparameter tuning
    of XGBoost, and all preprocessing required.
    """

    def __init__(self, train, test, mode='predict'):
        """
        Initialize train and test DataFrames
        :param train: path to train file
        :param test: path to test file
        :param mode: string indicating whether you want to produce predictions csv from Test.csv ('predict'),
                        run 10-fold with xgb ('xgb10'), tune xgb ('xgbtune'), tune adaboost ('adatune')
                        The latter 2 take a very long time!
        """
        self.train = pd.read_csv(train)
        self.test = pd.read_csv(test)

        if mode == 'predict':
            FinalIterationModel.train_and_predict(FinalIterationModel.preprocess(self.train),
                                                  FinalIterationModel.preprocess(self.test))
        elif mode == 'xgb10':
            scores = self.get_predictions_from_train(FinalIterationModel.preprocess(self.train), 10)
            print(scores)
            mean = statistics.mean(scores)
            print(mean)
        elif mode == 'xgbtune':
            FinalIterationModel.tune_booster(FinalIterationModel.preprocess(self.train))
        else:
            FinalIterationModel.tune_booster(FinalIterationModel.preprocess(self.train), 'ada')

    @staticmethod
    def preprocess(df):
        """
        Performs all preprocessing necessary for this iteration of our model; only columns we need are reviewText,
        summary, root-genre, and related. We do all processing and engineering here.
        :param df: DataFrame to be processed
        :return: processed DataFrame
        """
        df.drop('first-release-year', axis=1, inplace=True)

        # get review count or number of reviews
        df['review-count'] = 1
        df_review_count = df.groupby('amazon-id', as_index=False)['review-count'].sum()

        df['reviewText'].fillna('', inplace=True)
        df['summary'].fillna('', inplace=True)

        # VADER sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        df[['sum_neg', 'sum_neu', 'sum_pos', 'sum_compound']] = FinalIterationModel.get_vs(df['summary'], analyzer)
        df[['review_neg', 'review_neu', 'review_pos', 'review_compound']] = FinalIterationModel.get_vs(df['reviewText'],
                                                                                                       analyzer)

        df['reviewText'] = df['reviewText'] + ' ' + df['summary']  # simple unweighted concatenation for TextBlob

        # TextBlob sentiment analysis (used WITH VADER)
        df['review-polarity'] = df['reviewText'].apply(lambda s: TextBlob(s).sentiment.polarity)
        df['review-subjectivity'] = df['reviewText'].apply(lambda s: TextBlob(s).sentiment.subjectivity)

        df['reviewText'] = df['reviewText'] + ' ' + ((df['summary'] + ' ') * 4)  # weigh summary higher

        df['review-polarity'] = (df['review-polarity'] + 1) ** 2
        df['review-subjectivity'] = (df['review-subjectivity'] + 1) ** 2
        df_avg_pol = df.groupby('amazon-id', as_index=False)['review-polarity'].mean()
        df_avg_sub = df.groupby('amazon-id', as_index=False)['review-subjectivity'].mean()

        # sentiment analysis compound scores are scaled to be in range [1, 2] instead of [-1, 1], squared, then we take
        # mean for each product
        df_concat_text = df.groupby('amazon-id').reviewText.unique().agg(', '.join).reset_index()
        df['sum_compound'] = (df['sum_compound'] + 1) ** 2
        df_avg_summary = df.groupby('amazon-id', as_index=False)['sum_compound'].mean()
        df['review_compound'] = (df['review_compound'] + 1) ** 2
        df_avg_review = df.groupby('amazon-id', as_index=False)['review_compound'].mean()

        # drop all other cols effectively
        chosen_cols = df[['amazon-id', 'related', 'root-genre', 'target']]
        chosen_cols = chosen_cols.drop_duplicates(subset=['amazon-id'])

        final_df = pd.merge(chosen_cols, df_concat_text, on='amazon-id')

        final_df = pd.merge(final_df, df_avg_summary, on='amazon-id')
        final_df = pd.merge(final_df, df_avg_review, on='amazon-id')
        final_df = pd.merge(final_df, df_review_count, on='amazon-id')
        final_df = pd.merge(final_df, df_avg_pol, on='amazon-id')
        final_df = pd.merge(final_df, df_avg_sub, on='amazon-id')

        # we organized our code & progress code as best as we could so that everything we use in the final model
        # would be in this file, but buy_after_viewing was a method that we found increased scores in an earlier
        # iteration and then again in this final model. So we decided to not repeat code.
        final_df = DataPreprocessing.binarize_root_genre(final_df)
        final_df = DataPreprocessing.buy_after_viewing(final_df)
        final_df.drop('related', axis=1, inplace=True)

        print(final_df.columns)

        return final_df

    @staticmethod
    def get_vs(sentences, analyzer):
        """
        Gets sentiment analysis scores for our text
        :param sentences: a column in DataFrame with text data e.g. summary
        :param analyzer: instance of SentimentIntensityAnalyzer()
        :return: scores (multiple cols)
        """
        # neg, neu, pos, compound scores generated in that order
        l = len(sentences)
        scores = np.zeros((l, 4))
        print('start')
        for i, sentence in enumerate(sentences):
            vs = analyzer.polarity_scores(sentence)
            scores[i] = list(vs.values())

        return scores

    @staticmethod
    def prep_train_test(x_train, x_test, train_text, test_text, feature_names):
        """
        Prepares train and test DataFrames for fit and predict by concatenating a dense version of TF-IDF vectorizer's
        return sparse matrix
        :param x_train: train DataFrame
        :param x_test: test DataFrame
        :param train_text: sparse matrix produced by vectorizer for train data
        :param test_text: sparse matrix produced by vectorizer for test data
        :param feature_names: feature names provided by vectorizer
        :return: concatenated train and test DataFrames
        """
        train_text = pd.DataFrame(train_text.toarray(), columns=feature_names)
        x_train = pd.concat([x_train, train_text], axis=1)
        x_train.drop('reviewText', axis=1, inplace=True)

        test_text = pd.DataFrame(test_text.toarray(), columns=feature_names)
        x_test = pd.concat([x_test, test_text], axis=1)
        x_test.drop('reviewText', axis=1, inplace=True)

        return x_train, x_test

    @staticmethod
    def evaluate_xgb(x_train, y_train, x_test, y_test, model_name, estimators):
        """
        Evaluation method for XGB Learning API training and predicting--different from XGB SKLearn API (which we could
        have used the other evaluate method from Common with) but Learning API is significantly faster
        :param x_train: train feature space
        :param y_train: ground truth of train set
        :param x_test: test feature space
        :param y_test: ground truth of test set
        :param model_name: printed with f1 score and accuracy, for readability
        :return: weighted F1 score of prediction
        """
        d_train = xgb.DMatrix(x_train, label=y_train)
        d_test = xgb.DMatrix(x_test, label=y_test)
        watchlist = [(d_test, 'eval'), (d_train, 'train')]

        # tuned parameters
        params = {'objective': 'binary:logistic', 'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.8,
                  'colsample_bytree': 1, 'gamma': 1, 'seed': 42}
        clf = xgb.train(params, d_train, estimators, watchlist)

        y_pred = clf.predict(d_test)
        y_pred = np.round(y_pred)  # xgb gives probability of belonging to class 1 so we can round up

        f1 = f1_score(y_test, y_pred, average='weighted')

        print("f1 target", model_name, f1)
        print("acc target", model_name, accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        return f1

    @staticmethod
    def get_predictions_from_train(df, num_split, model_selection=False, tune_lr=False, model='xgb', estimators=-1):
        """
        Predicts 'target' column using 'reviewText', currently concatenation of 'summary' and 'reviewText', and
        'view_buy', and 'root-genre'.
        Contains code for model selection & hyperparameter tuning.
        :param df: full DataFrame
        :param num_split: k for k-fold cross validation
        :param model_selection: whether you want to run the code that prints performance of different models
        :param tune_lr: whether you want to run the code that tunes Logistic Regression's parameters
        :param model: string indicating whether you want to try Logistic Regression ('lr'),
                        Support Vector Machines ('svm'), AdaBoostClassifier ('ada'),
                        MLPClassifier ('mlp'), SGDClassifier ('sgd'), XGBoostClassifier ('xgb)
                        Default is 'xgb' because it's the final model we used
        :param estimators: n_estimators to use for Booster (ignored if not using Booster)
        """
        scores = []

        if model == 'lr':
            clf = LogisticRegression(solver='liblinear', penalty='l1')
        elif model == 'ada':
            if estimators != -1:
                clf = AdaBoostClassifier(n_estimators=estimators, learning_rate=0.5, random_state=42)
            else:
                clf = AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42)
        elif model == 'mlp':
            clf = MLPClassifier(solver='adam', hidden_layer_sizes=(13, 13, 13), random_state=42)
        elif model == 'sgd':
            clf = SGDClassifier(penalty='l1', random_state=42)

        cv = KFold(n_splits=num_split, random_state=42, shuffle=True)

        for train_index, test_index in cv.split(df):
            df_train, df_test = df.iloc[train_index, :], df.iloc[test_index, :]

            # get relevant columns only
            x_train = df_train[
                ['reviewText', 'review_compound', 'sum_compound', 'view_buy', 'root-genre', 'review-count',
                 'review-subjectivity', 'review-polarity']]
            y_train = df_train['target']
            x_test = df_test[
                ['reviewText', 'review_compound', 'sum_compound', 'view_buy', 'root-genre', 'review-count',
                 'review-subjectivity', 'review-polarity']]
            y_test = df_test['target']

            tfidf = TfidfVectorizer(analyzer='word', max_features=10000, ngram_range=(1, 2), token_pattern=r'\w{1,}')

            x_train_text, x_test_text = tfidf.fit_transform(x_train['reviewText']), tfidf.transform(
                x_test['reviewText'])

            # make sure indexes in train that are not actually removed from this DataFrame do not cause inf/-inf/nan
            # issues when we fit and predict
            x_train.reset_index(inplace=True)
            x_train_shrunk = x_train.copy(deep=True)
            x_train_shrunk.drop('index', inplace=True, axis=1)

            x_test.reset_index(inplace=True)
            x_test_shrunk = x_test.copy(deep=True)
            x_test_shrunk.drop('index', inplace=True, axis=1)

            feature_names = tfidf.get_feature_names()
            x_train_full, x_test_full = FinalIterationModel.prep_train_test(x_train_shrunk, x_test_shrunk, x_train_text,
                                                                            x_test_text, feature_names)

            if model_selection:
                # documentation of the model selection we did for this pipeline
                Common.evaluate_model(MultinomialNB(), x_train_full, y_train, x_test_full, y_test,
                                                   "multinomialnb")
                Common.evaluate_model(RandomForestClassifier(), x_train_full, y_train, x_test_full, y_test,
                                                   "randomforest")
                Common.evaluate_model(DecisionTreeClassifier(), x_train_full, y_train, x_test_full, y_test,
                                                   "decisiontree")
                Common.evaluate_model(KNeighborsClassifier(), x_train_full, y_train, x_test_full, y_test,
                                                   "kneighbors")
                Common.evaluate_model(GaussianNB(), x_train_full, y_train, x_test_full, y_test,
                                                   "gaussiannb")
                Common.evaluate_model(LogisticRegression(), x_train_full, y_train, x_test_full, y_test,
                                                   "logisticregression")

            # tunes params that significantly impact logistic regression's performance (we've already found that this
            # model works best for our data)
            if tune_lr:
                for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
                    for penalty in ['none', 'l1', 'l2', 'elasticnet']:
                        try:
                            Common.evaluate_model(LogisticRegression(solver=solver, penalty=penalty),
                                                               x_train_full, y_train, x_test_full, y_test,
                                                               solver + ' ' +
                                                               penalty)
                        except Exception as e:
                            # some penalties and solvers can't be used together from documentation, ignore
                            pass

            # evaluation function for Logistic Regression and ADABoost
            if model == 'lr' and clf is not None:
                f1 = Common.evaluate_model(clf, x_train_full, y_train, x_test_full, y_test, model)
            elif model == 'ada' and clf is not None:
                f1 = Common.evaluate_model(clf, x_train_full, y_train, x_test_full, y_test, model)
            elif model == 'mlp' and clf is not None:
                f1 = Common.evaluate_model(clf, x_train_full, y_train, x_test_full, y_test, model)
            elif model == 'sgd' and clf is not None:
                f1 = Common.evaluate_model(clf, x_train_full, y_train, x_test_full, y_test, model)
            else:  # evaluation with new method to take advantage of XGB's faster Learning API compared to sklearn API
                if estimators != -1:
                    f1 = FinalIterationModel.evaluate_xgb(x_train_full, y_train, x_test_full, y_test, model, estimators)
                else:
                    # default estimators is best one we found, for other params we tuned
                    f1 = FinalIterationModel.evaluate_xgb(x_train_full, y_train, x_test_full, y_test, model, 330)

            scores.append(f1)

        return scores

    @staticmethod
    def tune_booster(df, booster='xgb', folds=10):
        """
        Tunes booster, either ADABoost or XGBooster. Note that this function tunes n_estimators, but we tuned MANY
        MORE parameters. However, doing a grid search on these multiple parameters gave many memory errors and so
        basically never ended. Instead, we ran many times, changing different parameters one at a time, and deduced
        the best ones that way. Consider this as an example of the kind of tuning that we performed.
        :param df: full DataFrame
        :param booster: string indicating which booster to tune--AdaBoostClassifier ('ada') or default is 'xgb' because
                        it's the final model we used
        :param folds: number of folds for K-fold cross-validation
        """
        for estimators in range(250, 350, 10):
            FinalIterationModel.get_predictions_from_train(df, folds, model=booster)

    @staticmethod
    def train_and_predict(train, test):
        """
        Trains on train data set and produces a CSV of predictions on test data set (Test.csv).
        :param train: train DataFrame
        :param test: test DataFrame
        """
        # use tuned params
        tfidf = TfidfVectorizer(analyzer='word', max_features=10000, ngram_range=(1, 2), token_pattern=r'\w{1,}')

        x_train = train[['reviewText', 'review_compound', 'sum_compound', 'view_buy', 'root-genre', 'review-count',
                         'review-subjectivity', 'review-polarity']]
        y_train = train['target']
        x_test = test[['reviewText', 'review_compound', 'sum_compound', 'view_buy', 'root-genre', 'review-count',
                       'review-subjectivity', 'review-polarity']]

        train_text, test_text = tfidf.fit_transform(x_train['reviewText']), tfidf.transform(x_test['reviewText'])
        feature_names = tfidf.get_feature_names()

        x_train, x_test = FinalIterationModel.prep_train_test(x_train, x_test, train_text, test_text, feature_names)

        # fit model, predict test, then write predictions to CSV (path specified in string below)
        d_train = xgb.DMatrix(x_train, label=y_train)
        d_test = xgb.DMatrix(x_test)

        # tuned parameters
        params = {'objective': 'binary:logistic', 'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.8,
                  'colsample_bytree': 1, 'gamma': 1, 'seed': 42}
        clf = xgb.train(params, d_train, 330)

        y_pred = clf.predict(d_test)
        y_pred = np.round(y_pred).astype(int)  # convert to int (not actually necessary)
        output = pd.DataFrame({'amazon-id': test['amazon-id'], 'Awesome': y_pred})
        output.to_csv('final_predictions.csv')

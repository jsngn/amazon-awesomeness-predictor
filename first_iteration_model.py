import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score
import statistics
import numpy as np
from data_preprocessing import DataPreprocessing
from common import Common


class FirstIterationModel:
    """
    First iteration of our approach; we abandoned this approach because there were leakage issues with predicting
    whether a review is awesome (> 4 stars) using reviewText and summary, then using those predictions with other
    features in data to enhance predictions of whether a review is awesome. (We do some further processing, see
    documentation below, to judge whether a product is awesome given whether its reviews' are awesome).

    We split the train data set into a train and test set (instance variables below) in order to assess the impacts
    of our leakage, and needless to say, it was bad.

    See train_and_predict() documentation below for more details on this approach's pipeline.
    """
    def __init__(self, csv):
        """
        Reads a CSV file path and splits it in 2 for train and test set, to assess leakage's impact
        :param csv: file path of training data csv
        """
        df = pd.read_csv(csv)
        df.drop(df.columns[0], axis=1, inplace=True)
        dfs = np.array_split(df, 2)

        self.train = FirstIterationModel.preprocess_df(dfs[0])
        self.test = FirstIterationModel.preprocess_df(dfs[1])

    @staticmethod
    def preprocess_df(df):
        """
        Preprocess provided DataFrame according to our model's iteration
        :param df: provided DataFrame
        :return: DataFrame with relevant columns preprocessed
        """
        # every feature we use for this iteration is processed here; please see DataProcessing documentation for
        # further details on each method
        df = DataPreprocessing.review_binary(df)
        df = DataPreprocessing.binarize_root_genre(df)
        df = DataPreprocessing.cat_music(df)
        df = DataPreprocessing.unix_helpful_multiplier(df)
        df = DataPreprocessing.get_review_year(df)
        df = DataPreprocessing.unix_helpful_percent(df)
        df = DataPreprocessing.categorize_unix_helpful_percent(df)
        df = DataPreprocessing.buy_after_viewing(df)
        df['target'] = df['target'].apply(lambda x: True if x == 1 else False)
        df['helpful'] = df['helpful'].apply(DataPreprocessing.get_helpful_multiplier)

        # drop everything else, only use these
        df = df[
            ['amazon-id', 'reviewText', 'summary', 'helpful', 'review-binary', 'overall', 'target',
             'unix_help_multiplier', 'unix_help_percent_cat', 'year', 'root-genre', 'categories', 'view_buy']
        ]

        return df

    def get_text_score(self, col, max_features):
        """
        Gets predictions for review-binary using text data
        :param col: name of text column i.e. reviewText or summary
        :param max_features: max_features param for TFDIF Vectorizer
        :return: predictions for review-binary in 1D array
        """
        tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=max_features,
                                     ngram_range=(1, 2))
        lr = MultinomialNB()

        self.train.dropna(inplace=True)
        self.test.dropna(inplace=True)

        x_train, x_test, y_train, y_test = train_test_split(self.train[col], self.train['review-binary'],
                                                            test_size=0.2,
                                                            random_state=42)

        x_train, x_test = tfidf.fit_transform(x_train), tfidf.transform(x_test)
        lr.fit(x_train, y_train)

        print("multinomialNB score on train text:", lr.score(x_test, y_test))

        # Add labels back into train data frame; this is where leakage affected us badly because we are
        # using lr.predict on data which we have fit the model with 2 lines ago
        return lr.predict(tfidf.transform(self.test[col])), lr.predict(tfidf.transform(self.train[col]))

    def train_and_predict(self, model_selection=False):
        """
        Trains and predicts 'target' column in 4 stages:
            - Use TFDIF Vectorizer on summary and reviewText to predict review-binary
            - Use summaryTextScore and reviewTextScore and every other (cleaned) feature to predict review-binary
            - For each product, use mean of review-binary predictions to predict awesomeness i.e. target
            - Evaluate overall prediction
        :param model_selection: boolean whether you want to run code that evaluates different models
        """
        # reviewText processing
        test_review, train_review = self.get_text_score('reviewText', 10000)
        self.train['reviewTextScore'] = train_review
        self.test['reviewTextScore'] = test_review
        self.train.drop('reviewText', axis=1, inplace=True)
        self.test.drop('reviewText', axis=1, inplace=True)

        # summary processing
        test_summary, train_summary = self.get_text_score('summary', 6000)
        self.train['summaryTextScore'] = train_summary
        self.test['summaryTextScore'] = test_summary
        self.train.drop('summary', axis=1, inplace=True)
        self.test.drop('summary', axis=1, inplace=True)

        # split train/test data to predict target given cleaned features
        x_train, x_test, y_train, y_test = train_test_split(self.train[
                                                                ['reviewTextScore', 'summaryTextScore',
                                                                 'unix_help_multiplier', 'unix_help_percent_cat',
                                                                 'year', 'root-genre', 'categories', 'view_buy',
                                                                 'helpful']],
                                                            self.train['review-binary'], test_size=0.2, random_state=42)

        # documentation of the model selection we did for this pipeline
        if model_selection:
            Common.evaluate_model(GaussianNB(), x_train, y_train, x_test, y_test, "gaussian nb")
            Common.evaluate_model(DecisionTreeClassifier(), x_train, y_train, x_test, y_test, "decision tree")
            Common.evaluate_model(KNeighborsClassifier, x_train, y_train, x_test, y_test, "k neighbors")
            Common.evaluate_model(LogisticRegression(), x_train, y_train, x_test, y_test, "logistic regression")
            Common.evaluate_model(RandomForestClassifier(), x_train, y_train, x_test, y_test, "random forest")

        # we found that Decision Tree Classifier was the best one for this iteration
        # so use it on train data, which has been split into train and test sets themselves
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        print("decision tree classifier score on train data:", model.score(x_test, y_test))

        # get target predictions on actual test set, currently using DecisionTree
        train_pred = model.predict(self.test[
                                       ['reviewTextScore', 'summaryTextScore', 'unix_help_multiplier',
                                        'unix_help_percent_cat', 'year', 'root-genre',
                                        'categories', 'view_buy', 'helpful']
                                   ])
        self.test['target-pred'] = train_pred

        # get list of target-pred values for each product
        # (many reviews per product so should have different predictions)
        target_pred = {}
        for index, row in self.test.iterrows():
            if row['amazon-id'] not in target_pred:
                target_pred[row['amazon-id']] = []
            target_pred[row['amazon-id']].append(row['target-pred'])

        # final target prediction is rounded mean of all reviews' target predictions
        # 0.21 corrector improves score
        for id in target_pred:
            target_pred[id] = round(statistics.mean(target_pred[id]) - 0.21)

        # get 1D array of predictions corresponding to order of products in original DataFrame
        # then get classification report and weighted F1 using ground truth (data['target])
        truth = self.test[['amazon-id', 'target']].drop_duplicates(subset=['amazon-id'])
        preds = []
        for index, row in truth.iterrows():
            preds.append(target_pred[row['amazon-id']])
        print(classification_report(truth['target'], preds))
        print(f1_score(truth['target'], preds, average='weighted'))
        print(accuracy_score(truth['target'], preds))


m = FirstIterationModel('full_train.csv')
m.train_and_predict()

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from data_preprocessing import DataPreprocessing


class SecondIterationModel:
    """
    Second iteration of our approach. To avoid leakage, we employed TF-IDF as the only feature and trained a
    Logistic Regression model which we found outperforms other models, such as NB or KNN. The model predicts whether a
    review is awesome (has rating>4.5). Products with certain percentage of awesome reviews are marked awesome.
    Despite high accuracies and F1 scores, this approach is abandoned later because the optimal parameters for TF-IDF
    involve max_features=35k. The generated matrix is too large to be converted to a dense array and concatenated with
    other features (the jupyter notebook kernel always crashes).
    """
    def __init__(self, csv):
        # file should already have a target column
        self.df = pd.read_csv(csv)

        # too many nans in this column
        self.df.drop('first-release-year', axis=1, inplace=True)

        # drop empty rows
        self.df = self.df.dropna(axis=0)
        self.df = DataPreprocessing.review_binary(self.df)

        # concatenate reviewText and summary; weigh summary more by duplicating it 5 times
        # 5 is the best number we found
        self.df['text_sum'] = self.df['reviewText'] + (self.df['summary'] + ' ') * 5

    def train_and_predict(self):
        """
        Splits data into train and test set, then trains and predicts with Logistic Regression.
        Here we just use the vectorized text columns.
        """
        # randomly select products using numpy.choice, test_size=0.3
        # train and test set have 70% and 30% of unique amazon-id's respectively; each has rows of reviews belonging to
        # id's in X_train_ids and X_test_ids
        amazon_ids = self.df['amazon-id'].unique()
        X_train_ids = np.random.choice(amazon_ids, int(len(amazon_ids)*0.7), replace=False)
        X_test_ids = np.array([id for id in amazon_ids if id not in X_train_ids])

        X_train = self.df[self.df['amazon-id'].isin(X_train_ids)]
        X_test = self.df[self.df['amazon-id'].isin(X_test_ids)]
        
        # get the target of id's in train and test. since the targets of one id are the same, .mean() simply gives the
        # target value.
        y_train = X_train.groupby('amazon-id')['target'].mean()
        y_test = X_test.groupby('amazon-id')['target'].mean()

        # fit and transform TF-IDF matrices. see below for the tune_tfidf method for finding the best params employed
        # in the following line.
        tfidf = TfidfVectorizer(analyzer='word', max_features=35000, ngram_range=(1, 2), token_pattern=r'\w{1,}')
        X_train_text = tfidf.fit_transform(X_train['text_sum'])
        X_test_text = tfidf.transform(X_test['text_sum'])
        
        # train and predict model
        lr = LogisticRegression()
        lr.fit(X_train_text, X_train['review-binary'])
        y_text_pred = lr.predict(X_test_text)

        # match text_pred in the test set with test amazon-id
        temp = pd.concat((pd.DataFrame(X_test['amazon-id'].values), pd.DataFrame(y_text_pred)), axis=1)
        temp.columns = ['amazon-id', 'predicted']
        
        # use for-loop to find the optimal threshold
        # products with percentage of awesome reviews > threshold is marked awesome
        # the predictions are then compared with the target in the test set to produce f1 and accuracy
        # For logistic regression, 0.8 is the best threshold with f1 and acc near 0.77
        for i in np.arange(0.75, 0.9, 0.01):  # loop through values for threshold
            y_pred = temp.groupby('amazon-id')['predicted'].mean() > i
            
            # join pred and test on amazon-id
            final = pd.DataFrame(y_pred).merge(pd.DataFrame(y_test), on='amazon-id')
            
            # find f1 and accuracy
            f1 = f1_score(final['target'], final['predicted'], average='weighted')
            acc = accuracy_score(final['target'], final['predicted'])
            print(i, f1, acc)

    def tune_tfidf(self):
        """
        Find the optimal max_features parameter for TF-IDF, as well as other params (see below).
        :return:
        """
        # We ran multiple times to experiment with analyzer (default vs 'word'),
        # ngram_range ({1,1}, {1,2}, {2,2}, {1,3}, {2,3}), token_pattern (default vs r'\w{1,}') and
        # stopwords (default vs 'english').
        max_feats = np.arange(10000, 40000, 5000)
        for m_f in max_feats:
            tfidf = TfidfVectorizer(analyzer='word', max_features=m_f, ngram_range=(1, 2), token_pattern=r'\w{1,}')
            lr = MultinomialNB()
            x_train, x_test, y_train, y_test = train_test_split(self.df['reviewText'],
                                                                self.df['review-binary'],
                                                                test_size=0.2,
                                                                random_state=42)
            x_train, x_test = tfidf.fit_transform(x_train), tfidf.transform(x_test)
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            acc = accuracy_score(y_test, y_pred)
            print(m_f, f1, acc)


m = SecondIterationModel('full_train.csv')
m.train_and_predict()

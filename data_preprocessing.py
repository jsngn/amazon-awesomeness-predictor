import re
import nltk
import pandas as pd
from ast import literal_eval
import numpy as np
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer

from data_exploration import DataExploration


class DataPreprocessing:
    """
    Data preprocessing methods that we tried but most did not yield good results
    """

    def __init__(self, df):
        self.df = df

    @staticmethod
    def get_train_target(in_csv, out_csv):
        """
        Generate a csv file with avg-rating column (average rating of all reviews for a product) and target column
        (1 if product is awesome, 0 if not)
        :param in_csv: input CSV path
        :param out_csv: output CSV path
        """
        train_data = pd.read_csv(in_csv)

        # get mean of each review for every product
        avg_rating_df = train_data[['amazon-id', 'overall']].groupby('amazon-id').mean()
        full_train_data = pd.merge(train_data, avg_rating_df, how='inner', on='amazon-id')
        full_train_data = full_train_data.rename(columns={'overall_y': 'avg-rating', 'overall_x': 'overall'})

        # target is integer
        full_train_data['target'] = full_train_data.apply(lambda row: 1 if row['avg-rating'] > 4.5 else 0, axis=1)

        # many nulls in this col
        full_train_data.drop('first-release-year', axis=1, inplace=True)

        full_train_data.to_csv(out_csv)

    def get_labels_frequencies(self):
        """
        Gets frequency that values in 'label' column occur with target=1 (awesome) and target=0 (not awesome)
        """
        # dictionary of 'label' value to number of rows with that value and target = 1
        dict_label_to_positive_review = {}
        # dictionary of 'label' value to number of rows with that value and target = 0
        dict_label_to_negative_review = {}

        for index, row in self.df.iterrows():
            DataExploration.update_dict_frequency(row['label'], row['target'], dict_label_to_positive_review,
                                                  dict_label_to_negative_review)

        good_labels = {}
        for label in dict_label_to_negative_review:
            if dict_label_to_negative_review[label] > 0 and \
                    dict_label_to_positive_review[label] / dict_label_to_negative_review[label] > 5:
                good_labels[label] = None

        return good_labels

    def get_categories_frequencies(self):
        """
        Gets frequency that values in 'categories' column occur with target=1 (awesome) and target=0 (not awesome)
        """
        self.df['categories'] = self.df['categories'].apply(literal_eval)

        # dictionary of 'categories' value to number of rows with that value and target = 1
        dict_category_to_positive_review = {}
        # dictionary of 'categories' value to number of rows with that value and target = 0
        dict_category_to_negative_review = {}

        for index, row in self.df.iterrows():
            for category in row['categories']:
                DataExploration.update_dict_frequency(category, row['target'], dict_category_to_positive_review,
                                                      dict_category_to_negative_review)

        good_cats = {}
        for category in dict_category_to_negative_review:
            if dict_category_to_negative_review[category] > 0 and \
                    dict_category_to_positive_review[category] / dict_category_to_negative_review[category] > 4:
                good_cats[category] = None

        return good_cats

    @staticmethod
    def categorize_helper(x):
        """
        Categorize values of 'unix_help_percent' into manually selected thresholds
        :param x: value to be categorized
        :return: category
        """
        # We found using the crosstab that many of the non-5 star reviews were within a certain range
        # We therefore cut the data into two "boxes" that using the unixtime*helpful variable could be separated
        # These two boxes included 1/20th boxes which had 8% and 20-30% 5 star reviews, as opposed to 75-80% for the
        # reviews not in range
        if 0.0 < x < 2.154:
            return 1  # 8 percent
        elif 2.154 < x < 8.022:
            return 2  # 20 percent
        else:
            return 0

    @staticmethod
    def view_buy_convert(x, k):
        """
        Get length of an array that is nested within a dictionary; this is for processing 'related' column
        :param x: string to be parsed into dictionary
        :param k: key into dictionary to access the array
        :return: length of array
        """
        dx = literal_eval(x)

        if dx.get(k) is not None:
            return len(dx.get(k))
        return 0

    @staticmethod
    def weigh_review_score(row):
        """
        Returns the product of review-binary prediction using only reviewText, multiplied by helpful percent
        :param row: current row to perform multiplication
        :return: weighted score
        """
        return row['reviewTextScore'] * row['helpful']

    @staticmethod
    def weigh_summary_score(row):
        """
        Returns the product of review-binary prediction using only summary, multiplied by helpful percent
        :param row: current row to perform multiplication
        :return: weighted score
        """
        return row['summaryTextScore'] * row['helpful']

    @staticmethod
    def get_helpful_multiplier(x):
        """
        Returns a multiplier that corresponds to whether ratio of helpful ratings to total ratings is at least 0.48
        :param x: the string of form "[num1, num2]" to be parsed by literal_eval
        :return: 2 if denominator is 0 (from observations) or if ratio is at least 0.48, otherwise 1
        """
        x = literal_eval(x)

        if x[1] == 0:
            return 2
        else:
            if x[0] / x[1] >= 0.48:  # 0.48 is a tuned value
                return 2
        return 1

    @staticmethod
    def review_binary(x):
        """
        Returns boolean value for column review-binary, which is whether review > 4 stars
        :param x: current row in DataFrame
        :return: modified row with review-binary
        """
        x['review-binary'] = x['overall'] > 4
        return x

    @staticmethod
    def binarize_root_genre(x):
        """
        Manually categorized root-genre based on observations in data exploration
        :param x: DataFrame to modify
        :return: new DataFrame
        """
        cleanup_nums = {"root-genre": {"Pop": 0, "Rock": 0, "Classical": 0, "Latin Music": 0,
                                       "Country": 1, "Jazz": 0, "Dance & Electronic": 0, "Alternative Rock": 0,
                                       "New Age": 0,
                                       "Rap & Hip-Hop": 0, "Metal": 0, "Folk": 1, "R&B": 0, "Blues": 1, "Gospel": 1,
                                       "Reggae": 0}}
        x = x.replace(cleanup_nums)
        return x

    @staticmethod
    def cat_music(x):
        """
        Categorize music, which is simply increasing integers, which serve merely as a placeholder for their
        corresponding strings
        :param x: row to modify
        :return: modified row
        """
        x["categories"] = x["categories"].astype('category')
        x["categories"] = x["categories"].cat.codes
        return x

    @staticmethod
    def get_review_year(x):
        """
        Get review year for each review
        :param x: DataFrame to modify
        :return: modified DataFrame
        """
        x['year'] = x['reviewTime'].apply(DataExploration.review_year_convert)
        return x

    @staticmethod
    def unix_helpful_multiplier(train_data):
        """
        Combining unixReviewTime with helpful multiplier to explore possible model improvements
        :param train_data: DataFrame to modify
        :return: modified DataFrame
        """
        train_data['unix_help_multiplier'] = np.log(train_data['unixReviewTime']) * train_data['helpful'].apply(
            DataPreprocessing.get_helpful_multiplier)
        return train_data

    @staticmethod
    def unix_helpful_percent(x):
        """
        Combining unixReviewTime with helpful percent to explore possible model improvements
        :param x: DataFrame to modify
        :return: modified DataFrame
        """
        x['unix_help_percent'] = np.log(x['unixReviewTime']) * x['helpful'].apply(DataExploration.get_helpful_percent)
        return x

    @staticmethod
    def categorize_unix_helpful_percent(x):
        """
        Categorize values in unix_help_percent column into different 'bins'
        :param x: DataFrame to modify
        :return: modified DataFrame
        """
        x['unix_help_percent_cat'] = x['unix_help_percent'].apply(DataPreprocessing.categorize_helper)
        return x

    @staticmethod
    def buy_after_viewing(x):
        """
        Get length of buy_after_viewing list in related column
        :param x: DataFrame to modify
        :return: modified DataFrame
        """
        x['view_buy'] = x['related'].apply(lambda x: DataPreprocessing.view_buy_convert(x, 'buy_after_viewing'))
        return x

    @staticmethod
    def binarize_feature(local_train, local_test, column, ratio):
        """
        Counts the percentages of >4.5 reviews of each value in local_train[column]
        Those with percentages above the ratio are given 1, others 0.
        Examples of column can be 'label' or 'categories'
        :param local_train: train DataFrame
        :param local_test: test DataFrame
        :param column: column name, which is string
        :param ratio: threshold for binarizing
        :return: modified train and test DataFrames
        """
        threshold = ratio / (ratio + 1)
        good_feat = pd.DataFrame(local_train.groupby(column)['target'].mean() > threshold)

        local_train[column] = local_train[column].apply(lambda x: good_feat.loc[x])
        local_test[column] = local_test[column].apply(DataPreprocessing.binarize_helper, args=(good_feat,))

        return local_train, local_test

    @staticmethod
    def binarize_helper(x, good_feat):
        """
        Helper function for binarize_feature
        :param x: row to modify
        :param good_feat: DataFrame of 'good' features i.e. percentage is above ratio
        :return: result or False
        """
        if x in good_feat.index:
            return good_feat.loc[x, 'target']
        return False

    @staticmethod
    def custom_token(text):
        """
        Custom tokenizer for TF-IDF; results were not better when we tried using this
        :param text: text to apply the tokenizer on; TF-IDF handles calling and passing text
        :return: tokenized text
        """
        return re.sub(r'[^a-zA-Z]', ' ', text).split()

    @staticmethod
    def wn_lemmatize(text):
        """
        Wordnet lemmatizer/tokenizer for TF-IDF; results were not better
        :param text: text to apply the tokenizer on; TF-IDF handles calling and passing text
        :return: tokenized and lemmatized text with non-alpha characters removed
        """
        text = re.sub(r'\W', ' ', text)
        lm = WordNetLemmatizer()
        return [lm.lemmatize(word) for word in nltk.word_tokenize(text)]

    @staticmethod
    def get_text_blob_sentiment(df):
        """
        Get sentiment analysis using TextBlob library; modifies the DataFrame directly
        TextBlob has 2 components to sentiment analysis: polarity and subjectivity
        :param df: DataFrame whose 'reviewText' and 'summary' columns we want to apply TextBlob to
        """
        # get scores for reviewText only
        df['review-polarity'] = df['reviewText'].apply(lambda s: TextBlob(s).sentiment.polarity)
        df['review-subjectivity'] = df['reviewText'].apply(lambda s: TextBlob(s).sentiment.subjectivity)

        # get scores for summary only
        df['summary-polarity'] = df['summary'].apply(lambda s: TextBlob(s).sentiment.polarity)
        df['summary-subjectivity'] = df['summary'].apply(lambda s: TextBlob(s).sentiment.subjectivity)

        # get scores for summary concatenated with reviewText
        df['compound-polarity'] = (df['summary'] + ' ' + df['reviewText']).apply(lambda s: TextBlob(s).sentiment.
                                                                                 polarity)
        df['compound-subjectivity'] = (df['summary'] + ' ' + df['reviewText']).apply(lambda s: TextBlob(s).sentiment.
                                                                                     subjectivity)

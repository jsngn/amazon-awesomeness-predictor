import datetime
import pandas as pd
from scipy.stats.stats import pearsonr
from ast import literal_eval
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class DataExploration:
    """
    Class containing the collection of methods we used for exploring the data set
    Includes data visualization
    """

    def __init__(self, df):
        self.df = df

    @staticmethod
    def encode_root(x):
        """
        Assign a numerical label for each root-genre (no meaning behind the numbers' orders)
        :param x: DataFrame to modify
        :return: modified DataFrame
        """
        # Since we were using models such as the Decision Tree classifier, we decided to encode the features into more
        # readily valuable data. In this case we hard coded the individual root-genre characteristics as we found they
        # had a good correlation for whether a product was awesome or not. We at first coded them simply in order,
        # 1-16, but later re-ordered them according to their frequency/average scores. This is one of the only features
        # that we consistently found improved our model
        cleanup_nums = {"root-genre": {"Pop": 1, "Rock": 2, "Classical": 3, "Latin Music": 4,
                                       "Country": 5, "Jazz": 6, "Dance & Electronic": 7, "Alternative Rock": 8,
                                       "New Age": 9,
                                       "Rap & Hip-Hop": 10, "Metal": 11, "Folk": 12, "R&B": 13, "Blues": 14,
                                       "Gospel": 15,
                                       "Reggae": 16}}
        x = x.replace(cleanup_nums)
        return x

    @staticmethod
    def review_year_convert(x):
        """
        Get year that review was written from 'reviewTime' column
        :param x: element to parse and return
        :return: year as float
        """
        # We wanted to see if there was any pattern in the reviews from certain time periods
        # Indeed, we found that reviews past around 2013 were on average noticeably more positive (i.e. more 5 stars)
        year = x.split(', ')[1]
        if not year:
            return np.nan
        return float(year)

    @staticmethod
    def update_dict_frequency(tag, target, tag_to_positive_review, tag_to_negative_review):
        """
        Updates dictionaries that keep frequency of a row that has tag in a column (caller specifies) and a
        negative/positive target in 'target' column
        :param tag: value for this row of column whose relationship with negative/positive target we are interested in
        :param target: value of 'target' for this row
        :param tag_to_positive_review: dictionary of value : frequency
        :param tag_to_negative_review: dictionary of value : frequency
        """
        # We dug deep and use the individual review 'goodness' to estimate the 'awesomeness' of the product itself
        if tag not in tag_to_positive_review:
            if target == 1:
                tag_to_positive_review[tag] = 1
                tag_to_negative_review[tag] = 0
            else:
                tag_to_positive_review[tag] = 0
                tag_to_negative_review[tag] = 1
        else:  # just increment if tag is already in dictionary
            if target == 1:
                tag_to_positive_review[tag] += 1
            else:
                tag_to_negative_review[tag] += 1

    @staticmethod
    def get_helpful_percent(x):
        """
        Returns ratio in range [0, 1] of helpful ratings to total ratings of a review
        :param x: the string of form "[num1, num2]" to be parsed by literal_eval
        :return: -1 if denominator is 0, otherwise ratio as above
        """
        # We wanted to use the 'helpfulness' of a review to try to estimate its rating of the product
        # For this, we found a slight correlation with less helpful reviews being just slightly more negative
        x = literal_eval(x)

        if x[1] == 0:
            return -1
        else:
            return x[0] / x[1]

    def visualize_heat_map(self):
        """
        Graphs a heat map of correlations between different features in data set
        """
        # there is not a noticeable correlation between 'target' and any feature that wasn't calculated using 'overall'
        # (including 'target' and 'avg-rating'); we will need to do quite some processing
        sns.heatmap(self.df.iloc[:, 3:].corr())
        plt.show()

    def visualize_correlation_with_overall(self, feature):
        """
        For understanding correlation of a feature with 'overall' and 'binary_overall' (whether a review > 4 stars)
        Print correlation values and visualizes the distribution of feature vs 'overall' or 'binary_overall'
        :param feature: the name of the feature whose correlation we are interested in
        """
        randomness = np.random.rand(self.df.shape[0]) / 2 - 0.25
        
        # a review is awesome if its rating>4.5
        self.df['binary_overall'] = self.df['overall'] > 4.5

        print('corr with overall:', pearsonr(self.df['overall'], self.df[feature])[0])
        print('corr with binary_overall:', pearsonr(self.df['binary_overall'], self.df[feature])[0])
        
        # Just plotting different features to try to spot patterns
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].scatter(self.df['overall'] + randomness, self.df[feature], s=.01)
        axes[1].scatter(self.df['binary_overall'] + randomness, self.df[feature], s=.01)
        axes[0].set_title(f'overall vs {feature}')
        axes[1].set_title(f'binary_overall vs {feature}')

    def visualize_helpful(self):
        """
        Visualizes the impact of 'helpful' column on 'review-binary', which indicates whether a single review > 4 stars
        """
        self.df['helpful-percent'] = self.df['helpful'].apply(self.get_helpful_percent)

        # helpful categories: -1 if denominator == 0, 1 if ratio is at least 0.5, 0 otherwise
        self.df['helpful-cats'] = self.df.apply(
            lambda row: -1 if row['helpful-percent'] == -1 else (1 if row['helpful-percent'] >= 0.5 else 0), axis=1)

        # make review-binary which was a binary representation of whether an individual review was 5 stars or not
        self.df['review-binary'] = self.df['overall'] > 4

        # observations: helpful category 0 (i.e. there are some ratings but ratio is below 0.5) has similar number of
        # reviews > 4 stars and <= 4 stars, while the other 2 categories clearly indicate the review is > 4 stars
        # HOWEVER when we tried to predict 'target' using helpful, it does not improve results
        sns.countplot(x='helpful-cats', hue='review-binary', data=self.df)
        plt.show()

    def visualize_salesRank(self):
        """
        Visualizes the impact of 'salesRank' column on 'review-binary', which indicates whether a single
        review > 4 stars
        Requires 'review-binary' to be present in DataFrame
        """
        # We found that the more popular songs, in terms of sales-rank, had slightly higher ratings so we explored what
        # we could do with this sales rank categories based on different thresholds
        self.df['salesRank-cats'] = self.df.apply(
            lambda row: 0 if row['salesRank'] < 10000 else (1 if 10000 <= row['salesRank'] < 60000 else 2), axis=1)

        # observation: changed above thresholds to different values but we only observed that there are more > 4 star
        # reviews across the categories, no matter the threshold; when the threshold causes the category size to be
        # very small, the difference between > 4 star and <= 4 star review becomes less
        sns.countplot(x='salesRank-cats', hue='review-binary', data=self.df)
        plt.show()

    def visualize_price(self):
        """
        Visualizes the impact of 'price' column on 'review-binary', which indicates whether a single
        review > 4 stars
        Requires 'review-binary' to be present in DataFrame
        """
        # Generally we explored the price characteristic and found that it was an extremely poor predictor for anything
        # price categories based on thresholds
        self.df['price-cats'] = self.df.apply(lambda row: 1 if 28.41 < row['price'] <= 39.8 else 0, axis=1)

        # observation: adjusted above thresholds to different values but only observed that there are more > 4 star
        # reviews across all categories, as with salesRank
        sns.countplot(x='price-cats', hue='review-binary', data=self.df)
        plt.show()

    def visualize_unix(self):
        """
        Visualizes the impact of 'unixReviewTime' column on 'review-binary', which indicates whether a single
        review > 4 stars
        Requires 'review-binary' to be present in DataFrame
        """
        # We focused on the unix time as it gave us a granular view at the review level
        # We attempted to use this to increase the prediction of an individual review's 'awesomeness'
        # Here we are using the 2013 cutoff as previously, as it seemed to be somewhat indicative of a trend
        
        # unix review time binary categories, where reviews written on or after 2013 are new/1
        self.df['unix-cats'] = self.df['unixReviewTime'].apply(
            lambda x: 1 if datetime.datetime.fromtimestamp(x).year >= 2013 else 0)

        # observations: both categories seem to indicate higher chances of a review > 4 stars, as with salesRank and
        # price; this might not be useful because this feature could make all True predictions for review-binary
        # (or target); moreover this just seems to suggest that there are more > 4 star reviews across our data set,
        # rather than any relationship between year of review and whether review > 4 stars
        sns.countplot(x='unix-cats', hue='review-binary', data=self.df)
        plt.show()

    def print_labels_frequencies(self):
        """
        Prints frequency that values in 'label' column occur with target=1 (awesome) and target=0 (not awesome)
        """
        # dictionary of 'label' value to number of rows with that value and target = 1
        dict_label_to_positive_review = {}
        # dictionary of 'label' value to number of rows with that value and target = 0
        dict_label_to_negative_review = {}

        for index, row in self.df.iterrows():
            self.update_dict_frequency(row['label'], row['target'], dict_label_to_positive_review,
                                       dict_label_to_negative_review)

        # print the values whose ratio of positive : negative target > 5
        # observation: some labels lead to many more positive target than negative; can be useful
        for label in dict_label_to_negative_review:
            if dict_label_to_negative_review[label] > 0 and \
                    dict_label_to_positive_review[label] / dict_label_to_negative_review[label] > 5:
                print("label: " + label + "    positive/negative: " + str(
                    float(dict_label_to_positive_review[label] / dict_label_to_negative_review[label])))

    def print_categories_frequencies(self):
        """
        Prints frequency that values in 'categories' column occur with target=1 (awesome) and target=0 (not awesome)
        """
        self.df['categories'] = self.df['categories'].apply(literal_eval)

        # dictionary of 'label' value to number of rows with that value and target = 1
        dict_category_to_positive_review = {}
        # dictionary of 'label' value to number of rows with that value and target = 0
        dict_category_to_negative_review = {}

        for index, row in self.df.iterrows():
            for category in row['categories']:
                self.update_dict_frequency(category, row['target'], dict_category_to_positive_review,
                                           dict_category_to_negative_review)

        # print the values whose ratio of positive : negative target > 5
        # observation: some categories lead to many more positive target than negative; can be useful
        for category in dict_category_to_negative_review:
            if dict_category_to_negative_review[category] > 0 and \
                    dict_category_to_positive_review[category] / dict_category_to_negative_review[category] > 4:
                print("category: " + category + "    positive/negative: " + str(
                    float(dict_category_to_positive_review[category] / dict_category_to_negative_review[category])))

    def get_root_genre_overall_correlation(self):
        """
        Show the average review star for each root genre (encoded into integers)
        """
        self.df = self.encode_root(self.df)
        avg = self.df[['root-genre', 'overall']].groupby('root-genre').mean()
        # observation: categories with numbers 5, 12, 14, 15 (see encode_root above) have higher means than the rest;
        # we use this for manual binary encoding of root-genre
        print(avg.sort_values(by=['overall']))

    def get_crosstabs_per_review(self):
        """
        Shows distribution of values in 'overall' column in each bin of different columns
        What we did here was split the data into 'bins' and try to see whether we could isolate bad reviews 
        to improve our algorithm or spot patterns
        """
        # The price characteristic was generally usseless in predicting the score for individual reviews
        # It seemed almost like a wave, each bin having a slightly higher or lower average rating but not in any
        # noticeable linear pattern. It was this demonstration that caused us to drop this characteristic as being
        # something we wanted to use
        pd.crosstab(self.df['overall'], pd.qcut(self.df['price'], 10), normalize='columns')

        self.df['binary_overall'] = self.df['overall'] > 4  # same as review-binary in above functionss
        pd.crosstab(self.df['binary_overall'], pd.qcut(self.df['price'], 10), normalize='columns')

        # We did notice a pattern in root-genre and this is why this characteristic has continued to be useful
        pd.crosstab(self.df['overall'], self.df['root-genre'], normalize='columns')
        
        # There did seem to be a slight pattern here, with better sales (i.e. closer to being 1st ranked leading to
        # generally higher scores). The least selling of the products has some of the lowest average reviews
        # the last two bins of around 10k each had on average only 65% 5 star reviews, as opposed to average 71% over
        # the entire data set and slightly higher for the best ranked products
        pd.crosstab(self.df['overall'], pd.qcut(self.df['salesRank'], 10), normalize='columns')

        # We created a new variable to try to find whether the time between the year of release and the review year was
        # useful. Unfortunately, this variable did not end up being useful
        time_gap = self.df['reviewTime'].apply(self.review_year_convert) - self.df['first-release-year']
        self.df = self.df.drop('first-release-year', axis=1)
        self.df = self.df.dropna(axis=0)
        self.df['time_gap'] = time_gap
        pd.crosstab(self.df['overall'], pd.cut(self.df['time_gap'], 10), normalize='columns')

        pd.crosstab(self.df['overall'], self.df['label'], normalize=False)
    
        # Here we tried several things and were able to create a feature that combined both helpful data & time
        # We multiplied the log of reviewTime by the helpfulness of an individual review
        # we found that around the 0 mark in terms of this new variable, out of 500 reviews only 8% were 5 star reviews
        # we were really able to isolate the bad reviews and in our early algorithm this was really seen as a great way
        # to find the negative reviews. This combines the fact that early and late reviews are more positive, with the
        # trend that non-helpful reviews were also lower scores. This allowed unparalleled levels of isolation of
        # reviews
        self.df['helpunix'] = np.log(self.df['unixReviewTime']) * self.df['helpful'].apply(self.get_helpful_percent)
        pd.crosstab(self.df['overall'], pd.qcut(self.df['helpunix'], 20), normalize='columns')

    def get_crosstab_per_product(self):
        """
        Shows distribution of bins of salesRank vs product's average ratings.
        """
        # Whether we set bins=10 or 20, the lower rank a product has, the less likely it has a high average rating,
        # but salesRank does not prove useful when training models. We did not examine the distribution of
        # first-release-year because the feature has many nan values and some are even after the reviewTime,
        # thus it is judged as unreliable.
        gb = self.df.groupby('amazon-id')
        a = gb['overall'].mean()
        b = gb['salesRank'].mean()
        pd.crosstab(pd.cut(a, 5), pd.qcut(b, q=10), normalize='columns')

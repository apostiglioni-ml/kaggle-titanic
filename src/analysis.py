from functional import curried, compose
from functools import reduce
import numpy as np
import pandas as pd
from sklearn import tree, preprocessing, ensemble, neural_network, model_selection
from sklearn.model_selection import KFold, cross_val_score
from sklearn import feature_selection, pipeline
from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve

def aggregate(select, group_by, aggregate_fn, data):
    aggregated = data.pivot_table(select, index=group_by, aggfunc=aggregate_fn)
    def get_value(row):
        """For the given row, look up the aggregated value based on the 'group_by' value of the row.
           If the grouping value is null, then return null"""
        return aggregated.loc[row[group_by]][select] if pd.notnull(row[group_by]) else row[group_by]

    return data[[select, group_by]].apply(get_value, axis = 'columns')  # TODO: check if map can be used instead of apply

def fill_with(column, values, data):
    return data[column].mask(pd.isnull, values)

def fill_with_aggregate(select, group_by, aggregate_fn, data):
    return fill_with(select, aggregate(select, group_by, aggregate_fn, data), data)

def fill_with_mode(column, data):
    mode = data[column].mode().at[0]
    return data.fillna({ column: mode })

def fill_age(data):
    rounded_mean = compose(round, np.mean)
    return data.assign(Age = fill_with_aggregate(select='Age', group_by='Title', aggregate_fn=rounded_mean, data=data))

def fill_embarked(data):
    return fill_with_mode('Embarked', data)

def estimate_family_size(df):
    return df['Parch'] + df['SibSp'] + 1

def add_family_size(data):
    return data.assign(FamilySize = estimate_family_size(data))

def extract_family_name(data):
    return data['Name'].str.extract('([A-Za-z]+),', expand = False)

def add_family_name(data):
    return data.assign(FamilyName = extract_family_name(data))

def extract_title(data):
    return data['Name'].str.extract(' ([A-Za-z]+)\.', expand = False);

def add_title(data):
    return data.assign(Title = extract_title(data))

def add_cabin_size(data):
    return data.assign(CabinSize = aggregate(select='PassengerId', group_by='Cabin', aggregate_fn='count', data=data)) \
               .fillna({'CabinSize': 1})                                                                               \
               .astype({'CabinSize': np.int32})

def add_ticket_size(data):
    return data.assign(TicketSize = aggregate(select='PassengerId', group_by='Ticket', aggregate_fn='count', data=data))

def estimate_group_size(data):
    get_max = lambda row: max([row['FamilySize'], row['CabinSize'], row['TicketSize']])

    return data.apply(get_max, axis='columns')

def categorize_ages(df, bins=8, precision=0):
    return pd.cut(df['Age'], bins=bins, precision=precision, right=False)

def add_age_group(df):
    return df.assign(AgeGroup = categorize_ages(df))

def add_group_size(data):
    return data.assign(GroupSize = estimate_group_size(data))

def sex_and_class(df):
    return df[['Sex', 'Pclass']].apply(lambda x: '@'.join(map(str,x)), axis='columns')

def add_sex_and_class(df):
    return df.assign(Sex_Class = sex_and_class(df))

def estimate_ticket_price(data, outlier_low, outlier_high):
    calculate_group_size = compose(add_group_size, add_family_size, add_cabin_size, add_ticket_size)

    ticket_prices = data.assign(TicketPrice = data['Fare'] / calculate_group_size(data)['GroupSize'])
    outliers = (ticket_prices['TicketPrice'] <= outlier_low) | (ticket_prices['TicketPrice'] > outlier_high)

    return ticket_prices[~outliers][['Pclass', 'TicketPrice']].groupby('Pclass').agg(['min', 'max', 'mean'])

def estimate_mean_fare(df, outlier_low, outlier_high):
    prices = estimate_ticket_price(df, outlier_low=outlier_low, outlier_high=outlier_high)

    get_price = lambda row: row['GroupSize'] * prices.loc[row['Pclass']]['TicketPrice']['mean']
    return df[['Pclass', 'GroupSize']].apply(get_price, axis='columns')

@curried
def fill_fare(df, outlier_low=None, outlier_high=None):
    return df.assign(Fare = fill_with('Fare', estimate_mean_fare(df, outlier_low, outlier_high), df))

def bounded_fare(df, outlier_low, outlier_high):
    calculate_group_size = compose(add_group_size, add_family_size, add_cabin_size, add_ticket_size)

    ticket_prices = df.assign(TicketPrice = df['Fare'] / calculate_group_size(df)['GroupSize'])
    outliers = (ticket_prices['TicketPrice'] <= outlier_low) | (ticket_prices['TicketPrice'] > outlier_high)

    bounds = ticket_prices[~outliers][['Pclass', 'TicketPrice']].groupby('Pclass').agg(['min', 'max'])

    get_price = lambda agg: lambda row: row['GroupSize'] * bounds.loc[row['Pclass']]['TicketPrice'][agg]
    bounded_max = df[['Pclass', 'GroupSize']].apply(get_price('max'), axis='columns')
    bounded_min = df[['Pclass', 'GroupSize']].apply(get_price('min'), axis='columns')
   
    fares = fill_fare(outlier_low=outlier_low, outlier_high=outlier_high)(df)['Fare']

    return (fares.mask(ticket_prices['TicketPrice'] >= outlier_high, bounded_max)
                 .mask(ticket_prices['TicketPrice'] <= outlier_low, bounded_min))

@curried
def add_bounded_fare(df, outlier_low=0, outlier_high=60):
    return df.assign(BoundedFare = bounded_fare(df, outlier_low=outlier_low, outlier_high=outlier_high))

@curried
def drop_columns(columns, data):
    return data.drop(columns, axis='columns')

@curried
def survival_rate(by, df):
    survived = df.groupby([by, 'Survived']).size()
    return (survived.groupby(level=by)
                    .apply(lambda x: x / float(x.sum()))    # Calculate percentage
                    .unstack('Survived')
                    .fillna(0)                              # NaN appears in cases where either all survived or died
                    [[1,0]])                                # Reorder columns, first survived

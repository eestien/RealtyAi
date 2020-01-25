from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

DATA_TERM = 'C:/Storage/DDG/DATA/PREPARED/COORDINATES_PREP_DATA.csv'

def DataObservation():

    data = pd.read_csv(DATA_TERM)




    # X and Y correlation
    # sns.scatterplot(x='price', y='full_sq', data=outliers_it)
    # plt.show()



    # Distibution
    # sns.distplot(data['price'])
    # plt.ylim(0, 60000)
    # plt.xlim(0, 50000000)
    # plt.show()


    # Generate a Boxplot
    # data['price_meter_sq'].plot(kind='box')
    # plt.show()

    # Generate a Histogram plot
    data['latitude'] = np.log1p(data['latitude'])
    data['latitude'].plot(kind='hist')
    plt.show()

    # data['price_meter_sq'] = np.log(data['price_meter_sq'] + 1)
    # plt.hist(data['price_meter_sq'], color='blue')
    # plt.show()


    # MODEL FEATURE IMPORTANCES
    # feature_importance = clf.feature_importances_
    # make importances relative to max importance
    # feature_importance = 100.0 * (feature_importance / feature_importance.max())
    # list_of_features = ['renovation', 'has_elevator', 'longitude', 'latitude', 'price', 'full_sq', 'kitchen_sq',
    #                     'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y', 'price_meter_sq',
    #                     'clusters']
    # print(list(zip(list_of_features, feature_importance)))
    '''
    gbr = load("C:/Storage/DDG/PRICE_AND_TERM/Models/GBR_PRICE.joblib")
    cat = load("C:/Storage/DDG/PRICE_AND_TERM/Models/CAT_GRADIENT.joblib")
    data['pred_price'] = data[
        ['renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
         'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y', 'clusters']].apply(
        lambda row:
        int(((np.expm1(gbr.predict([[row.renovation, row.has_elevator, row.longitude, row.latitude, row.full_sq,
                                     row.kitchen_sq, row.is_apartment, row.time_to_metro, row.floor_last,
                                     row.floor_first, row.X, row.Y, row.clusters]])) + np.expm1(
            cat.predict([[row.renovation, row.has_elevator, row.longitude, row.latitude, row.full_sq,
                          row.kitchen_sq, row.is_apartment, row.time_to_metro, row.floor_last,
                          row.floor_first, row.X, row.Y, row.clusters]])))[0] / 2)), axis=1)

    data['profit'] = data[['pred_price', 'price']].apply(
        lambda row: (((row.pred_price * 100 / row.price) - 100) * 100), axis=1)
    # DATA CORRELATION
    print(data.corr().term.sort_values(ascending=False))
    '''

    # Plotting clusters
    # sns.lmplot('Lattitude', 'Longtitude', data = df2, fit_reg=False,hue="clusters",
    # scatter_kws={"marker": "D", "s": 100})

    # Plot distribution of flats based on price and quantity
    # x_labels = list(range(0, 201000000, 5000000))
    # plt.hist(data['price'], bins=x_labels)
    # plt.xticks(x_labels)
    # plt.xticks(rotation=90)
    # plt.xlim([-2, max(x_labels)])
    # plt.savefig('E:/Realty_sk/distr.jpg')
    # plt.show()


DataObservation()

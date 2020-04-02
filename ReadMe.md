Uses Python 3.7

Installed packages:
* Sklearn
* Pandas 
* Numpy
* Joblib (to load the model)
* SciPy
* pip install backports-datetime-fromisoformat






- - - -
### Profitable Offers ###
_Profitable offers finds via output result of stacking price models. If predicted price higher than real - will assume, that it is profitable offer. 

__Preprocessing__

1. `DATA_PREP_MOSCOW.py`, `DATA_PREP_SPB.py` - data preprocessing. Using different csv files to concatenate to one main csv. 
Creating KMeans model for clustering flats by coordinates.

2. `DataVisualize.py` - some useful functions to plot/visualize data

__Main__
 
`PRICE_SPB.py`, `PRICE_MOSCOW.py` - train price regression models.
Stacking algorithms: Gradient Boosting machine regression, RandomForest Regressor, Light GBR.

### Run Price and Sale Time prediction ###
1. `app.py` - Run.  server receives request with parameters and 
returns the predicted price based on parameters. Price

Prediction is based on the parameters transmitted with the request.
Agorithm stacking is used: Gradient Boosting machine regression from sk-learn and lgbm regression, Random forest regression.

__Flats features using for price and sale time prediction__

1. Price Prediction:

Prediction based on 12 parameters('life_sq', 'rooms', 'renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
              'time_to_metro', 'floor_last', 'floor_first', 'clusters''). 

2. Sale Time Prediction:

Prediction based on 14 parameters('price', 'profit', 'life_sq', 'rooms', 'renovation', 'has_elevator',
 'longitude', 'latitude', 'full_sq', 'kitchen_sq', 'time_to_metro', 'floor_last', 'floor_first', 'clusters')
  

### PRICE PREDICTION ###

### Data preprocessing: ###
                           
__1. Log Transformation `np.log1p(something)` for price(target label), longitude, latitude, rooms, life_sq, full_sq, kitchen_sq:
 The Log Transformation can be used to make highly skewed distributions less skewed.
 The comparison of the means of log-transformed data is actually a comparison of geometric means. 
 This occurs because the anti-log of the arithmetic mean of log-transformed values is the geometric mean.__

 

### Plotting price  - sale time  correlation for a few prices ###

The main idea is visualize (and predict) different sale time for different flat prices. I.e. if (as proposed) 
flat is more profitable(sale price is less than it must be) it will sell out faster. And vice versa. 

### Оценка стоимости объекта и срока продажи. ###

![Screenshot](https://github.com/eestien/RealtyAi/blob/master/screen_example.png)

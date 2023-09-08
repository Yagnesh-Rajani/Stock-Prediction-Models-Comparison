# Predicting Stock Market Trends Using Predictive Analytics - Capstone Summary
Capstone project created by Yagnesh Rajani as a part of the "Data Analytics, Big Data, and Predictive Analytics" certificate offered by the Chang School of Continuing Education (which is part of the Toronto Metropolitan University). This study was supervised by Professor Ceni Babaoglu.

## Abstract
The United States Stock Markets have an impact on both the local economy and the global economy and predicting stock prices has become an important capability for day-traders, fund managers, companies, banks, and other investors. Over the last several years, Machine Learning systems have been growing in capabilities including analyzing data for important features and creating forecasts based on those features. This purpose of this capstone project is to apply Machine Learning models to determine important features for predicting stock prices based on market data, generate price predictions, determine which Machine Learning models are most accurate in predicting stock prices by analyzing the correlation between the predicted prices and actual prices, and investigating how the models respond to the impact of the COVID-19 pandemic. The dataset being used originates from Kaggle which tracks stock prices in the New York Stock Exchange (NYSE) from the beginning of 1980 to present date and contains a variety of trading data and indicators for each stock. Using the Python programming language, the data was preprocessed, analyzed for important features, used to create time series forecasting models which were visualized and analyzed for performance. The time series forecasting models used are Auto Regressive Integrated Moving Average (ARIMA), Seasonal Auto Regressive Integrated Moving Average (SARIMA), and Prophet. This study found that the most accurate forecasting model for both the standard and COVID sets was the ARIMA model based on Root Mean Squared Error value.

## Exploratory Data Analysis

### Data Description
The first step was to acquire the dataset which I did from Kaggle.com (https://www.kaggle.com/datasets/footballjoe789/us-stock-dataset?resource=download). The dataset consists of Comma Separated Files on 7297 stocks in the NYSE and contains a large number of trading information such as the date, opening price, high, low, closing price, trading volume, dividends and stock splits. The dataset also contains technical analysis indicators such as the Relative Strength Index, Commodity Channel Index, Chande Momentum Oscillator, etc. The data for each stock is stored in its own Comma Separated File. For the purposes of this thesis, 1 stock was selected based on its Morningstar performance (Hicks 2023) as well as the length of the dataset. The stock selected is Walt Disney Co which holds the “DIS” symbol on the NYSE and holds data from January 2, 1962 to May 12, 2023 when the dataset was acquired for analysis. The raw dataset can be found in the "DIS.csv" file.

### Data Cleaning
The data was then cleaned by removing unwanted columns and formatting the data. The attributes selected were the Date, Open, High, Low, Close, Volume, Dividends and Stock Splits. Most of the selected columns were of an acceptable format however the "Date" column was changed from an object format to a datetime64 format. This data was saved to the "dataset.csv" file as formatting specific to each algorithm would be done before the algorithm was run.

### Data Visualization
The Open, High, Low and Close attributes have similar looking plots which can be expected due to the nature of the market. On the scale of the graphs, even a drastic rise or drop in price would be unnoticeable. What can be seen however is the movement of the market over time which grows slowly for until 2008 after which is takes a much more aggressive upwards trajectory until 2020 where the market shows massive growth followed by almost as much of a crash. The Volume attribute has a much different graph over time and while there is growth in the peaks of trading volume there is no massive spike in volume after 2020. The only noticeable spike is in the mid 1980s which is likely due to the Black Monday crisis. The Dividends attribute also shows growth in its peaks which generally track the movement of the market while the Stock Splits attribute shows only a few occasions where the stock is split. 

![image](https://github.com/Yagnesh-Rajani/Stock-Prediction-Models-Comparison/assets/129909709/27a28e00-b329-4f30-af2a-f29b641c7099)


Another method of visualization of data is the use of boxplots as seen in below and were created using the same loop method as the line plots were however this time, using the seaborn library to create the visuals.

![image](https://github.com/Yagnesh-Rajani/Stock-Prediction-Models-Comparison/assets/129909709/f7e70966-b351-45ee-a304-701a9d2e91b9)


The boxplots of each attribute show a similar picture for the Open, High, Low and Close attributes as they have similar quartile and whisker values and somewhat similar outlier 
patterns. The Volume attribute also shows a number of outliers however the majority of the data is much more concentrated as the outliers are significant. Similarly, the Dividends attribute has a small concentration of data with a few outliers which is expected as the majority of Dividends were 0 with a few exceptions. This is also reflected in the Stock Splits Attribute however much less frequently as there were only a few splits. All together, there are a large number of outliers for the majority of attributes.

### Outliers
Outliers present a problem when conducting a time series analysis and can be split into two different types of outliers, each with a different set of actions to be taken to manage them. The first type of outliers are actual outliers occurring rarely in the dataset. These can impact modelling and must be handled through removal or replacement with a determined value such as the median value. The second type of outliers are an important part of the dataset due to significant movement of the data however they are presented as outliers as there are significantly more datapoints which are higher or lower. To determine which of the two the outliers in the dataset fall into, a custom function called outlier_detector was developed. The function takes in a set of data and a list of attributes, cycles through the data of each attribute to calculate the quartile values and the interquartile range as well as whisker values of each attribute, and then compares the data to the whiskers and stores any outliers in a list which can 
be seen below.

![image](https://github.com/Yagnesh-Rajani/Stock-Prediction-Models-Comparison/assets/129909709/fd15d6b6-8b04-4727-8cad-1419165fd2dd)


To understand if the outliers were a portion of the data and not the occasional outlying value, the range of the outliers was found. The earliest outlier was on December 30, 2013 and the last outliers was on May 12, 2023 with 2344 values being outliers. This strongly suggests that the outliers were not occasional values but a significant part of the movement of the data. This is further confirmed in above where most of the attributes had strong upwards movement in values from 2010 onwards. As the dataset begins in 1962, the majority of the data were much smaller values compared to those found during the aggressive upwards movement of the last decade. This confirms that the outliers are not the occasional value which needs to be managed but instead a natural part of the data.

### Correlation

Correlation is measured on a scale of -1 to 1. A negative value represents a negative correlation with -1 representing a perfectly negative correlation. A positive value represents a positive correlation with 1 representing a perfectly positive correlation. A correlation value of 0 represents perfectly uncorrelated data. Finding the correlation between attributes is fairly simply in Python as a function can be applied to the dataset containing the selected attributes and the results are displayed as a seaborn heatmap below.

![image](https://github.com/Yagnesh-Rajani/Stock-Prediction-Models-Comparison/assets/129909709/c55ffd29-dd6e-4456-916f-619ff93d7021)


The correlation values depict an interesting picture of the relationships between the attributes. There is a strong positive correlation between the Date attribute and the Open, High, Low and Close attributes which is understandable as the values of these attributes tends to grow over time especially since 2010 as discussed above. There is a weak positive correlation between the Volume attribute and the Open, High, Low and Close attributes as well as between the Date attribute and the Volume attribute. There is a near perfect positive correlation between the Open, High, Low and Close attributes which reflects the very similar line plots above. Lastly, the remaining correlation values are very close to 0 suggesting practically no correlation.

### Seasonality Analysis
A seasonal decomposition of each attribute is shown in the image below. This was done using the seasonal_decompose function of the statsmodel library. For each model, the trend matches the dataset closely although there is some seasonality and residual values. As a result, a seasonality of 4 will be used for the SARIMA model.

![image](https://github.com/Yagnesh-Rajani/Stock-Prediction-Models-Comparison/assets/129909709/93864376-a57a-4de2-826d-f338835cdabf)


## Time Series Forecasting

### Common Components
Each model requires different code to run however there are some areas of overlap, namely the preparation of data for analysis. This common portion of each model is explained below.

In order to prepare the data for analysis, the following steps were taken using Python:
1. The data was loaded into the model and attributes specified.
2. The Date attribute was formatted to fit the requirements of the model.
3. An evaluation function was set up to calculate the Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). The calculations are largely handled by the sklearn metrics library.
4. The data was split into a training set consisting of 80% of the data and a testing set which contained the remaining 20% of the data. The date of the split is February 3, 2011. This model will be called the Standard Model.
5. The data was split into a COVID-19 training set consisting of all the data until January 2, 2020 and a testing set consisting of the remaining data. This model will be called the COVID Model.

### ARIMA
ARIMA is the first time series forecasting model used in this capstone and it is based on regression analysis. It is made up of three parts which are the Autoregression (AR), Integrated (I) and Moving Average (MA) components. The AR component is used to model a changing variable that regresses on its own prior values. The Integrated component takes raw data and uses differencing to make it stationary. The Moving Average component determines how dependent an observation and residual error are using a moving average model.

Please see the "ARIMA.ipynb" file to view the code and visuals generated. The visuals for the "Open" attribute are shown below to provide a quick understanding of the results.

##### Chart Displaying ARIMA Standard Model Open Prediction
![image](https://github.com/Yagnesh-Rajani/Stock-Prediction-Models-Comparison/assets/129909709/eef38b3e-5fc6-4a0b-a125-76d4f2034b51)

##### Chart Displaying ARIMA COVID Model Open Prediction
![image](https://github.com/Yagnesh-Rajani/Stock-Prediction-Models-Comparison/assets/129909709/6f530dfa-e53e-49fd-9385-144a7848018d)

#### ARIMA Model Conclusions
The predicted standard models for the Open, High, Low, Close and Volume attributes was fairly similar in shape with an initial drop and then a steady flat line. The initial drop is negligible on the scale of the entire dataset as well as the scale of the training data which gives the appearance of a straight line. This suggests that the models may adopt a wave instead of a line however due to the scale of the graph this cannot be seen. None of the models were close to the training dataset however which suggests that the training data may not include enough variation for the model to predict the data more accurately. As mentioned above, much of the outliers of the data are within the last decade of the data which would lie in the testing dataset which explains why the model predicts a low value with relatively no variation. This is somewhat confirmed in the COVID models for the same attributes where model shows much more variation and different movements of the model.

For the Dividends dataset, the standard model portrays a wave shape before settling which is also reflected in the COVID model. This makes sense as there is little variation within the data for the attribute and any movement would be a significant jump from the normal value.

The Stock Splits model predicted higher values than the testing data showed. This is likely due to most of the splits occurring before the standard model’s split date.

### SARIMA
The second model is SARIMA model or Seasonal ARIMA model. It extends the ARIMA model to include seasonal components. In addition to the AR, I and MA components mentioned above, the SARIMA model has a seasonal AR, seasonal I and seasonal MA components and a seasonality component.

Please see the "SARIMA.ipynb" file to view the code and visuals generated. The visuals for the "Open" attribute are shown below to provide a quick understanding of the results.

##### Chart Displaying SARIMA Standard Model Open Prediction
![image](https://github.com/Yagnesh-Rajani/Stock-Prediction-Models-Comparison/assets/129909709/ee06bbe3-e402-40ec-aef9-3f5655aaedf9)

##### Chart Displaying SARIMA COVID Model Open Prediction
![image](https://github.com/Yagnesh-Rajani/Stock-Prediction-Models-Comparison/assets/129909709/87882394-4877-4f47-8d08-7abc3dd8b61b)

#### SARIMA Model Conclusions
The predicted standard models for most of the attributes was fairly similar with there being a small upwards or downwards movement before flattening into a what looks like a line but is likely extremely small oscillations with a few exceptions. The first exception is the Close attribute which showed significant oscillating movement with large initial variation between highs and lows which settles to a much smaller variation. Similarly, the Dividends attribute shows the same type of movement however the oscillations shorten in height rapidly. On the scales of the dataset or training data however, the variation is invisible.

The COVID models show straight line movement with no oscillations even on a small scale and it is very far below the testing data in most models. This suggests that the inclusion of seasonality as well as the increased training data which has a lot of relatively passive movement compared to the overall data played a large role in shaping not only the variation of the models but also lowering the predicted values.

### Prophet
The final model is the Prophet developed by Facebook. It is an additive model which fits nonlinear trends with different stages of seasonality along with holiday effects. The stages are yearly, weekly and daily.

Please see the "Prophet.ipynb" file to view the code and visuals generated. The visuals for the "Open" attribute are shown below to provide a quick understanding of the results.

##### Chart Displaying Prophet Standard Model Open Prediction
![image](https://github.com/Yagnesh-Rajani/Stock-Prediction-Models-Comparison/assets/129909709/7371d854-13c4-4897-af05-d1da91e65197)

##### Chart Displaying Prophet COVID Model Open Prediction
![image](https://github.com/Yagnesh-Rajani/Stock-Prediction-Models-Comparison/assets/129909709/57fe75c1-e7c7-4334-a59f-e587aa9bd86d)

#### Prophet Model Conclusions
The Prophet Covid Models are noticeably better at predicting the stock data than the Standard models are. This is most likely due to the increased amount of training data with aggressive variation. This is not as noticeable when observing the Volume, Dividends and Stock Splits attributes as there is significantly less variation compared to the Open, High, Low and Close attributes. In all models the variation of the COVID models were much smaller than the Standard models which suggests that the model is far more confident in its predictions.

## Results
The MSE value is used to show the mean value of the squares of the variation. In order to ensure that errors do not cancel each other out, the variation is squared and the average is calculated to return a standard value for the entire dataset. The key issue with MSE is that the units are also squared, so to resolve this the square root of the MSE value is taken to standardize the units resulting in the RMSE value. The RMSE value is used to compare the variation between a model and the actual data. With both MSE and RMSE, the smaller the value the better as it shows that there is less variation. One drawback of using MSE and RMSE is that very large or small values or significant large or small values can skew the resulting value away from the median. The MAE accounts for this skew this by giving each variation an equal weighting and returns an average absolute error value. The MAPE value is the MAE value as a percentage. All of these values were calculated in the interests of having different metrics to value the performance of each model, however for the purposes of this study the RMSE will be used as variation is an important factor when assessing the accuracy of the models. Based on the RMSE values, the most accurate model for each attribute can be seen in the table below.

![image](https://github.com/Yagnesh-Rajani/Stock-Prediction-Models-Comparison/assets/129909709/d61b8c57-6919-4ecd-a78b-4facb5bf9028)


Additionally for all models, the COVID model had better RMSE values for the Open, High, Low and Close attributes whereas the standard models had better or very similar RMSE values for the Volume, Dividends and Stock Splits attributes.

## Conclusion
In conclusion, this study was designed to assess the performance of different time series forecasting models in both a standard model design as well as a COVID model design. The data was formatted and each model was run and assessed. Overall, the ARIMA model was the most accurate in predicting values with the least variation. The SARIMA model was the most accurate in a predicting a few attributes and the Prophet model was the most accurate in predicting two attributes. As the ARIMA model was the most common model with the best RMSE value for both the standard and COVID models, it is the best model to implement.

Additionally, the COVID models were more accurate in predicting most of the attributes. This is likely due to the increase in training data which contained more variation than the standard data did.

## Criticisms and Continuity
There are several criticisms to be made about this study and they are as follows:
1. The first is that Autocorrelation Function, Partial Autocorrelation Function tests were not conducted to give parameters for the ARIMA and SARIMA models. This study relied heavily on the auto_arima function of the pmdarima library to do a lot of this work.
2. An Augmented Dickey-Fuller test or similar test was not conducted to assess stationarity of the dataset. This means that potentially important information about the dataset and handling it may have been missed.
3.  While the seasonal value was assumed to be 4 based on observation, different seasonality’s should be tested. One drawback is that the larger the seasonality, the more computing resources and time is required especially for the SARIMA models.

The study can be continued in the following ways:
1. With increased seasonality comes the need for better hardware. This presents the opportunity to use cloud systems however it will require a lot of resources and time to run the models using a higher seasonality value.
2. Analyzing the difference between Open and Close prices as well as High and Low prices and running models on those values as they can present important information and may show interesting correlations.
3. Implement the missed tests mentioned above and use that information in running the models as well as optimizing the order and seasonal order values.

# Notes and Experience:
I learned quite a bit from this project as it differed from the machine learning algorithms I had worked with through the various courses in the "Data Analytics, Big Data, and Predictive Analytics" certificate and I had to learn the fundamentals of this topic and how to apply them. Aside from teaching myself the fundamentals of time series analysis, I had to learn how to apply the different algorithms to my dataset.

The next key area of learning was that, unlike decision trees and naive-bayes algorithms, time series analysis takes a long time and a lot more computational resources than I had anticipated while researching, teaching myself and setting up the project. If I was to do this project again or continue it, I would likely look into using some form of cloud systems to run this project if only to retain the use of my computer for other uses. I would likely start with a high RAM Linode server (or Akamai server, I believe Linode is being merged with Akamai) but I would need to research other cloud providers and pricing before I make a decision.

The final key area of learning was that my study is much different from a lot of other studies and resources I used and this led to a lot of time being wasted. As you can see above, many of the graphs generated for the models have relatively straight lines. This caused a lot of confusion and rework for me as I did not understand why my results seemed to only produce a straight line while online tutorials, videos and even other studies had variations in their forecasts. The answer to this, as I found out, was because of time and scale. Most other resources produced forecasts from anything between a few hours to maybe a few years. My Standard model forecast was some 12 years long with the training set being around 49 years long and my COVID model was around 2.5 years long with the training set being around 58 years long. The COVID model was longer than most resources I had come across and the Standard model had no similar studies that I could find. This meant that I was comparing the movement of relatively short term predictions to my long term predictions. Additionally, the training sets had a huge impact on the predictions, especially the Standard model. For the Standard model, 49 years of training with relatively small variation (compared to the variation in the testing data) lead to what seems to be a very low and straight forecast which makes sense since the model would predict slow growth and variation based on its training - hence the seemingly underperforming prediction. A similar story is made for the COVID model however we can see that it does not seem to underperform as much due to the additional variation in training gained in the longer training set (and the increased variation over the last decade or so compared to the decades before that). On the scale of the graphs, short term predictions are visible but predictions as long as mine show no movement because the variation of the predictions is simply too small to be noticeable - even when the forecast is isolated the sheer time scale smooths out the movement which is where I was having the disconnect between my results and those found in the online resources and studies. As a result, I spent a significant amount of time making small fixes and running the models for hours when I did not otherwise need to and faced a large time crunch. This time would have been better used to ensure my models were producing the correct and best possible results through proper testing (as opposed to the aformentioned time wasted on trying to fix things that did not need fixing).

There is no purpose to make forecasts of this length other than for the sake of learning, which was the main reason for completing this capstone study. Forecasting 12 years into the future based solely on previous performance has little value since it does not take into account new trends, technologies, financial regulations, etc. That is to say, there is no further information fed into the algorithm to help it predict future values aside from previous values that do not have any significance from each other.

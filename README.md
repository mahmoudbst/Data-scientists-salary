# Data-scientists-salary
- Predicting the data scientist salary based on glassdoor jobs (MAE ~ $ 11K) to help data scientists negotiate their income when they get a job.
- Engineered features from the text of each job description to quantify the value companies put on.
- Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model.

## Code and Resources Used
- Packages: pandas, numpy, sklearn, matplotlib, seaborn, pickle
- Scraper Article: https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905


## Data Cleaning
After getting the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:

- Parsed numeric data out of salary
- Made columns for employer provided salary and hourly wages
- Removed rows without salary
- Parsed rating out of company text
- Made a new column for company state
- Transformed founded date into age of company
- Made columns for if different skills were listed in the job description:
- Column for simplified job title and Seniority
- Column for description length

## EDA
I looked at the distributions of the data and the value counts for the various categorical variables.
![](Capture%20d’écran%202021-10-10%20232222.png)
![](Capture%20d’écran%202021-10-10%20232310.png)
![](Capture%20d’écran%202021-10-10%20232347.png)


## Model Building
First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.

I tried three different models:

- Multiple Linear Regression – Baseline for the model
- Lasso Regression – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
- Random Forest – Again, with the sparsity associated with the data, I thought that this would be a good fit.


## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets.

- Random Forest : MAE = 11.22
- Linear Regression: MAE = 18.86
- Lasso : MAE = 19.67

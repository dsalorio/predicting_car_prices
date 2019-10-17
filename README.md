# predicting_car_prices

Various regularized regression models for predicting car prices, using web scraped data

## Project Description and Inspiration

Unless you walk onto the lot knowing you're going to buy a BMW or a Tesla (and can somehow afford to do so), buying a car is not fun. You're confounded by multiple brands, countless options, and the fact that a test drive imparts only the barest sense of what a car is actually like.

So it's hard to know what you're paying for, what a car is realistically worth, and relatedly, how to spot a deal. Part of the inspiration for this project is understanding how car pricing works. The other part is illustrating a regression model workflow and champion model selection.

The data I use is all scraped from the www.thecarconnection.com, a website that compiles technical specs on cars and then has professional automotive journalists test drive and rate them. In addition to the ratings they provide MANY categories of technical details. Most of the esoteric add-ons, like fog-lights, aren't specified for most cars, so from a modeling standpoint they aren't useful, but out of about 235 features and about 32,000 cars there's a lot of data to work with.

The workflow is
* Load technical data on ~32,000 cars and 234 features from a csv (previously scraped)
* Clean and prepare this data, then scrape and add ratings to the dataframe
* Perform feature exploration and engineering, looking at feature correlations, pairplots, and using a simple linear regression. Select ln(price) as the predicted value to correct for non-linearity in the dataset
* Use k-folds cross-validation to find the optimal alpha (sometimes called lambda) values for Ridge and Lasso regularized linear regressions
* Select the champion model, a Lasso regression, since R<sup>2</sup>s are similar across all models, but Lasso is simpler to imterpret because it eliminates features. Use it to generate a test statistic on the holdout data, and finally refit it using the whole dataset

Since the scraper takes a few hours to gather the ratings data I've included a csv of the ratings data. It has the same index as the technical specs data, so you can just merge them on the index. Assuming you've already used the code in the notebook to load and prepare the fullspecs.csv data, you can just skip the webscraping section entirely and get right to modeling with this line of code.
```python
ratings = pd.read_csv('ratings.csv', index_col=0)
cars_df = cars_df.merge(ratings, how='inner', left_index=True, right_index=True)
```

## Results and Discussion

The Lasso Regression model has an R<sup>2</sup> of 0.93 on out-of-sample data, so it actually explains quite a lot of the variation in prices, and it eliminates quite a few of the extraneous features in the initial regression. Interestingly, it eliminates Rating as a feature, suggesting that rating doesn't actually have much of an impact on price. If you're a price sensitive buyer, this is really good news, because it means you should be able to get a car that automotive journalists like without spending a lot of money. So maybe it's not worth springing for that Tesla or BMW. Anyway, as someone once told me not long after buying an expensive car, "it was pretty cool having it at first, but now it's just how I get to work". Fair enough.

The features with the greatest impact on price are weight and horsepower, which are somewhat correlated, powerful engines being larger and heavier. Car brand also has a notable impact on car price, with non luxury brands having a discrete drop in price over more expensive marks. The takeaway is that you can get a deal on big, powerful cars (if that's your thing) by buying Fords and Chevys.

## Future Work

Ratings had surprisingly little impact on price in this model, so I'd like to model rating as a function of technical specs to understand it better. I would expect that relationship to be fairly direct, but perhaps design quality and overall driving experience are not dependent on components and can be achieved inexpensively.

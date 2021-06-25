# Predicting Speed Dating Outcomes

## Overview

In this project we conduct exploratory data analysis to narrow down and clean features for prediction via machine learning.

The data we work with represents four minute dates between individuals during speed dating events between the years of 2002 and 2004. A total of twenty-two waves were held and participants filled out surveys before, during, and after their dates.

![alexander-sinn-KgLtFCgfC28-unsplash.jpg](https://raw.githubusercontent.com/ismizu/Phase_3_Project/main/images/alexander-sinn-KgLtFCgfC28-unsplash.jpg)

<center>
    Image by
    <a href = https://unsplash.com/@swimstaralex>Alexander Sinn</a>
    from
    <a href = https://unsplash.com>Unsplash</a>
</center>

## Data Structure

All notebooks can be found in the /notebooks folder.
- Data insights can be found in the Business_Insights notebook
- EDA and modeling can be found in the Modeling notebooks


Data can be found in the /data folder. Obtained from [Kaggle.com](https://www.kaggle.com/annavictoria/speed-dating-experiment)


All images can be found in the /images folder.


[Business Insights Presentation](https://docs.google.com/presentation/d/1D584urKrCWcNV2nkgPIlVxCe3ZPuJ7dHYZYQXkUSxmQ/edit?usp=sharing)

Contributors:
- [Isana Mizuma](https://github.com/ismizu)
- [Tyrell Jackson](https://github.com/Tyronious25)

## Primary Focus

Our primary focus is to:
1. Evaluate our data
2. Clean our data into a useable format
3. Compare/contrast models to narrow down the best choice

<h1 align = 'center'> Business Analysis </h1>

One of the first items we looked into was the distribution of stated goals in participants, split by gender.

![goals.png](https://raw.githubusercontent.com/ismizu/Phase_3_Project/main/images/goals.png)

Looking at the graph above, we see that the distribution is generally the same between both sexes. 

With a large majority, attending the event because it seemed fun has an almost equal percentage. In addition, both sexes had their goal as meeting new people as the second highest rated goal although women did have a notably higher percentage in this area than men.

However, one notable difference is in those who stated their goal as "Get a Date." Seeing this imbalance, we take a closer look and find that just over twice as many men are looking to date as compared to women.

When we look at the percentage of those looking for a date, we find that the percent of men is more than twice that when compared to women.

For a speed dating event, there is a surprisingly low percentage of individuals actually looking for a date. In addition, there is a large imbalance between the sexes.

As such, we take a look at how the sexes rate the following:
- What they stated they looked for in potential dates
- What they actually went for in potential dates

![stated_vs_actual.png](https://raw.githubusercontent.com/ismizu/Phase_3_Project/main/images/stated_vs_actual.png)

Within the stated interests, we can see that men rated attraction as the trait they most look for, with intelligence coming in second. Women rated the attributes relatively equally with intelligence, fun, and sincerity rating the highest overall.

However, when we look at the actual interests the ratings change quite a bit. On the right side we graph how men and women felt about their actual matches. Despite the way both men and women stated their interests, attraction came out as, by far, the most influential factor in their initial decision.

<h1 align = 'center'> Modeling Process </h1>

After data cleaning, we begin some preliminary modeling. Primarily, a base logistic regression model to serve as our base. The first step is to investigate feature correlation with our target 'dec'

![base_model_perform.png](https://raw.githubusercontent.com/ismizu/Phase_3_Project/main/images/base_model_perform.png)

Our base model shows a 64% accuracy in predicting that an individual would want to date another. 

This means that our model can only correctly identify an individual's interest in dating 64% of the time.

With this score, and the low correlation of our features with the target variable, we take a look at a few other models.

![all_base.png](https://raw.githubusercontent.com/ismizu/Phase_3_Project/main/images/all_base.png)

Looking at the scores, there is some improvement in scores. However, the models', particularly KNearestNeighbor, hyperparameters are causing quite a bit of overfitting. We will have to revisit both the parameters as well as the data to fix this issue.

However, we do note that the scores did not improve by much, leading us to believe that continued cleaning/tuning may not result in the .75 accuracy we are looking for. Running Random Forest with grid search resulted in a .70 accuracy, however the recall dropped quite a bit, showing that it was more accurate at predicting 0's, but fell in its predictions for 1's.
For our predictions, we are placing a heavier weight into predicting 1's, or yes', so this model may not be feasible as well.

At this point, we decided to take a look at how the date itself influences ones decision, rather than predictions based solely on pre-date factors.

# Modeling with Post Date Survey Data

![during_date.png](https://raw.githubusercontent.com/ismizu/Phase_3_Project/main/images/during_date.png)

Using only the eight columns containing how the individual rated their partner during the four minute date, we were able to achieve exactly the .75 we were looking for.

As defined in our Business Case notebook, depending on gender, there were notable differences between what an individual stated they wanted, and what how they rated partners that they identified positively with. We can see this in the model predictions. Despite our best efforts, our accuracy remains quite low when only utilizing pre-date information.

With this, we realize that what is most important in bringing the two individuals together in the first place. Due to stark differences in what an individual believes they want compared to how they view those same traits during the date, a pre-date prediction performs minimally better than flipping a coin.


Knowing this, we take one more look at the data. This time, we explore the effect of physical attraction as our investigation into various insights showed this mattered greatly, regardless of whether the individual rated it as important or not.

# Physical Attraction based Prediction

After isolating physical attraction, we run the model to view its effects.

![all_bases.png](https://raw.githubusercontent.com/ismizu/Phase_3_Project/main/images/all_bases.png)

Using only the physical attraction that the individual felt, we are able to get very close to our prediction based on all during-date attribute ratings and we are not too far off from our both_dataframes_base_model.

As such, we come to the conclusion that one of the most important aspects to engaging with another individual is perceived physical attraction.

Blind dating applications where individuals cannot see each other before engaging may be met with limited success as a result. What is most important is that individuals have the chance to view each other through profile pictures, galleries, etc and have the chance to judge their physical attraction to each other.

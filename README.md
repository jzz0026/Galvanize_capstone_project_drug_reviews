# Sentiment Analysis of prescription drugy reviews and key words of side effects extraction
## Motivation
- This is the first business project hold by Good Grug Medical, a startup data analysis company.  The dataset used in this project provides patient reviews on specific drugs along with related conditions and a 10 star patient rating reflecting overall patient satisfaction. The data was obtained by crawling online pharmaceutical review sites  (Druglib.com). 
- Sentiment Analysis of these reviews could be useful for pharmacy companies and doctors to quickly get into bad reviews and know patients' complains. Also, a lot of symptoms and drug sides were hiden under reviews, which will be great to be automatically extracted to improve the drug and help to give a better prescription. 
- We try to utilize this dataset to do data mining for patient behavior and drug effectiveness in order to provide useful information to patients, doctors and pharmacy companies.

## Target
- Find diseases with significantly seasonal variations based on dates of posting comments.  
- Train the model to predict sentiment of reviews as “Positive”, “Negative”, and “Neutral” based on vectorized words from patients’ comments.
- Explore side effects from patients’ comments.

## EDA
- The number of comments posted had a trend of increasing with time seires. The website probably became more popular after 2015. 
<p align="center">
   <img src="Plot/num_comments_date.png" alt="alternate text">
</p>
- The number of comments for each condition/disease was unbalanced. We had a extremely high number of comments for birth control.
<p align="center">
   <img src="Plot/num_comments_each_condition.png" alt="alternate text">
</p>

## Find conditions/diseases with significantly seasonal variations
- The figour below showed the percentage of comments for the top 50 conditions with the most comments among 12 months. Several of them obviously presented some preference on distinct months, such as weight Loss and cough.	
<p align="center">
   <img src="Plot/Per_comments_month.png" alt="alternate text">
</p>



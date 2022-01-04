# 50.007 Machine Learning Project
Many start-up companies are interested in developing automated systems for analyzing sentiment information associated with social media data. Such sentiment information can be used for making important decisions such as making product recommendations, predicting social stance and forecasting financial market trends.
The idea behind sentiment analysis is to analyze the natural language texts typed, shared and read by users through services such as Twitter and Weibo and analyze such texts to infer the users’ sentiment information towards certain targets. Such social texts can be different from standard texts that appear, for example, on news articles. They are often very informal, and can be very noisy. It is very essential to build machine learning systems that can automatically analyze and comprehend the underlying sentiment information associated with such informal texts.
In this design project, we would like to design our sequence labelling model for informal texts using the hidden Markov model (HMM).

## How to run the approach codes
### Part 1:
```
python HMM_p1.py
```
### Part 2/3:
```
python HMM_p23.py
```
### Part 4:
```
python HMM_p4.py
```

## How to run evaluation script
### Part 1:
```
python evalResult.py ES/dev.out ES/dev.p1.out
python evalResult.py RU/dev.out RU/dev.p1.out
```
### Part 2:
```
python evalResult.py ES/dev.out ES/dev.p2.out
python evalResult.py RU/dev.out RU/dev.p2.out
```
### Part 3:
```
python evalResult.py ES/dev.out ES/dev.p3.out
python evalResult.py RU/dev.out RU/dev.p3.out
```
### Part 4:
```
python evalResult.py ES/dev.out ES/dev.p4.out
python evalResult.py RU/dev.out RU/dev.p4.out
```
For test data,
```
python evalResult.py <gold standard> ES/test.p4.out
python evalResult.py <gold standard> RU/test.p4.out
```
Replace the *<gold standard*> with your gold standard for evaluation, path should be included.

# 50.007 Machine Learning Project

In this design project, we would like to design our sequence labelling model for informal texts using the hidden Markov model (HMM) that we have learned in class. We hope that your sequence labelling system for informal texts can serve as the very first step towards building a more complex, intelligent sentiment analysis system for social media text.

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

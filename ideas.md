# Strategies

0. EDA
1. Preprocessing
    - Rankgauss, normalization (can ensemble these)
2. Stack models as much as possible
    - lstm, xgboost, catboost, rnn, dae, etc
    - ensemble after each prediction prior to the others
3. Try to generate further insights
    - Does the 25m, 6m, and 500k populations have correlations or data leakages between them
4. Pretraining and weights initialization
    - How does one model apply to another task?
    - Try lining them up end to end maybe.
5. Augmentation
6. Reverse engineering the creation (since it is generative)
7. Visualizing the output of the model.

# Possibly dumb ideas
- GAN (generative)
- Classifier
- Label smoothing
- Scale all of the datasets to be similar
- Don't think pseudolabelling works but could try it
- NN stack
- Postprocessing
- CTGAN
- Classifier/regressor for peak height, normalize all of the heights individually
- Infer between datapoints (into hours)
- SWI
- add a feature called "unvaccinated population"

# Models
- DAE
- Tabnet
- XGB
- Catboost
- LGBM
- LSTM
- RNN
- Rapids (svm, rf)
- Ridge
- Logistic
- NN/MLP
- https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/222745
- Naive/seasonal naive model
- Seasonal decomposition
- ARIMA (AutoRegressive Integrated Moving Average)
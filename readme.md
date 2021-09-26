# DCP COVID-19 Challenge

## "Building a model is very simple, but building a simple model is the hardest thing there is"

24 hours, 1700 lines of code, and 74 models, and here we are. This was so much fun, thanks to the organizers.

Given more time, we would have loved to simplify this process; but with the time we were given, a massive ensemble was the way to go.
### Our task 1 solution
- Stack models (in total, 74 models)
    - Diverse ensemble consisting of many types of models trained with different schemes (see below).
- Stack after each single prediction so that following predictions are made on more accurate data. We stack with a weighted average based on val MSE
- Robust cross validation. For cross validation, we use 3 folds, one for each CSV. We scale each CSV individually so that this is possible.
- Feature engineering for some models. Using meta labels, such as cumulative case counts, total vaccinated, etc. we get much better MAE, faster convergence, and our model does not "lag" one day behind in our CV; this indicates better robustness.

Meta features: `infected_unvaccinated  infected_vaccinated  total_vaccinated  days_increasing  cumulative_cases`
Would have liked to use `eligible_infections` but due to not having access to total populations, we elected to eliminate that feature.

Unfortunately we did not have time to tune our models at all, but here is a summary.

Models:
- LSTM with/without meta
- GRU with/without meta
- CNN with/without meta
- XGB without meta
- MLP without meta

![](https://cdn.discordapp.com/attachments/746585161067397236/891555866271186944/unknown.png)
Above is a model trained with metadata and a lag of 5 days (predicting/training 5 days into the future). Note that this is out of fold, so the model has not trained on this data yet.

![](https://cdn.discordapp.com/attachments/746585161067397236/891524987234689054/unknown.png)
Similar model scheme on a different fold

![](https://cdn.discordapp.com/attachments/746585161067397236/891720040930488420/unknown.png)
Graph of our submission 1 (days 300-400 are predicted)

![](https://cdn.discordapp.com/attachments/746585161067397236/891722505243480084/unknown.png)
Graph of our submission 2 

![](https://cdn.discordapp.com/attachments/746585161067397236/891723658098262057/unknown.png)
Graph of our submission 3
### Usage
1. `pip install -r requirements.txt`
2. Then import task 2 `from our_sub import model_prediction`.
    - For example, `model_prediction('./input/observations_1.csv', 100)`
    - Saves to `predictions.csv`

### Code explanation
`pipeline` contains the major pipeline components; training individual models, and inferencing an ensemble.
- `full_inference.py` does inference and ensembles models; it supports models both utilizing metadata, and pure time series.
- `full_train.py` is the main file for our task 2 submission; it trains an LSTM and a GRU on the feature engineered dataset. While we would have loved to train our entire suite of models, the time it takes makes it difficult to justify.
- `tf_folds.py` contains the underlying training code for our LSTM, GRU, etc; all non-gradient boosting models
- `xgb_baseline.py` contains our underlying XGBoost regressor training code.

Our rudimentary feature engineering and other preprocessing is in `utils/preprocess.py`.

### Table of all 74 models, and OOF MSE
| path                                           | mse   | graph overlap? | type |
| ---------------------------------------------- | ----- | -------------- | ---- |
| ./models/tf\_fold\_0\_gru\_nostack\_nometa.h5  | 80559 | y              | gru  |
| ./models/tf\_fold\_1\_gru\_nostack\_nometa.h5  | 3458  | y              | gru  |
| ./models/tf\_fold\_2\_gru\_nostack\_nometa.h5  | 3653  | y              | gru  |
| ./models/tf\_fold\_0\_gru\_stackp1\_nometa.h5  | 67975 | y              | gru  |
| ./models/tf\_fold\_1\_gru\_stackp1\_nometa.h5  | 3304  | y              | gru  |
| ./models/tf\_fold\_2\_gru\_stackp1\_nometa.h5  | 3713  | y              | gru  |
| ./models/tf\_fold\_0\_gru\_stackp1\_meta.h5    | 67938 | n              | gru  |
| ./models/tf\_fold\_1\_gru\_stackp1\_meta.h5    | 2869  | n              | gru  |
| ./models/tf\_fold\_2\_gru\_stackp1\_meta.h5    | 3575  | n              | gru  |
| ./models/tf\_fold\_0\_gru\_nostack\_meta.h5    | 73331 | n              | gru  |
| ./models/tf\_fold\_1\_gru\_nostack\_meta.h5    | 3404  | n              | gru  |
| ./models/tf\_fold\_2\_gru\_nostack\_meta.h5    | 3620  | n              | gru  |
|                                                |       |                |      |
| ./models/tf\_fold\_0\_lstm\_nostack\_nometa.h5 | 67555 | y              | lstm |
| ./models/tf\_fold\_1\_lstm\_nostack\_nometa.h5 | 3406  | y              | lstm |
| ./models/tf\_fold\_2\_lstm\_nostack\_nometa.h5 | 3650  | y              | lstm |
| ./models/tf\_fold\_0\_lstm\_stackp1\_nometa.h5 | 68388 | some           | lstm |
| ./models/tf\_fold\_1\_lstm\_stackp1\_nometa.h5 | 3406  | some           | lstm |
| ./models/tf\_fold\_2\_lstm\_stackp1\_nometa.h5 | 3650  | some           | lstm |
| ./models/tf\_fold\_0\_lstm\_stackp1\_meta.h5   | 65502 | n              | lstm |
| ./models/tf\_fold\_1\_lstm\_stackp1\_meta.h5   | 3092  | n              | lstm |
| ./models/tf\_fold\_2\_lstm\_stackp1\_meta.h5   | 3544  | some           | lstm |
| ./models/tf\_fold\_0\_lstm\_nostack\_meta.h5   | 73155 | n              | lstm |
| ./models/tf\_fold\_1\_lstm\_nostack\_meta.h5   | 3255  | n              | lstm |
| ./models/tf\_fold\_2\_lstm\_nostack\_meta.h5   | 3658  | n              | lstm |
|                                                |       |                |      |
| ./models/tf\_fold\_0\_mlp\_nostack\_nometa.h5  | 74789 | y              | mlp  |
| ./models/tf\_fold\_1\_mlp\_nostack\_nometa.h5  | 2893  | badly          | mlp  |
| ./models/tf\_fold\_2\_mlp\_nostack\_nometa.h5  | 3686  | y              | mlp  |
| ./models/tf\_fold\_0\_mlp\_stackp1\_nometa.h5  | 69837 | y              | mlp  |
| ./models/tf\_fold\_1\_mlp\_stackp1\_nometa.h5  | 3777  | y              | mlp  |
| ./models/tf\_fold\_2\_mlp\_stackp1\_nometa.h5  | 3675  | y              | mlp  |
|                                                |       |                |      |
| ./models/tf\_fold\_0\_cnn\_nostack\_nometa.h5  | 60457 | y              | cnn  |
| ./models/tf\_fold\_1\_cnn\_nostack\_nometa.h5  | 3968  | y              | cnn  |
| ./models/tf\_fold\_2\_cnn\_nostack\_nometa.h5  | 3669  | y              | cnn  |
| ./models/tf\_fold\_0\_cnn\_stackp1\_nometa.h5  | 69113 | y              | cnn  |
| ./models/tf\_fold\_1\_cnn\_stackp1\_nometa.h5  | 3487  | y              | cnn  |
| ./models/tf\_fold\_2\_cnn\_stackp1\_nometa.h5  | 3599  | y              | cnn  |
| ./models/tf\_fold\_0\_cnn\_stackp1\_meta.h5    | 42393 | what           | cnn  |
| ./models/tf\_fold\_1\_cnn\_stackp1\_meta.h5    | 2446  | what           | cnn  |
| ./models/tf\_fold\_2\_cnn\_stackp1\_meta.h5    | 3465  | what           | cnn  |
| ./models/tf\_fold\_0\_cnn\_nostack\_meta.h5    | 40150 | what           | cnn  |
| ./models/tf\_fold\_1\_cnn\_nostack\_meta.h5    | 3673  | what           | cnn  |
| ./models/tf\_fold\_2\_cnn\_nostack\_meta.h5    | 3501  | what           | cnn  |
|                                                |       |                |      |
| ./models/xgb\_fold\_0\_nostack.pkl             | 70719 | y              | xgb  |
| ./models/xgb\_fold\_1\_nostack.pkl             | 2906  | y              | xgb  |
| ./models/xgb\_fold\_2\_nostack.pkl             | 3588  | y              | xgb  |
| ./models/xgb\_fold\_0\_stackp1.pkl             | 66489 | y              | xgb  |
| ./models/xgb\_fold\_1\_stackp1.pkl             | 2527  | y              | xgb  |
| ./models/xgb\_fold\_2\_stackp1.pkl             | 3571  | y              | xgb  |

### What we wish we tried
1. Augmentations. We could have spliced certain aspects of one city with the other cities
2. Clipping results of the model using some sort of model - perhaps ARIMA
3. Even more models, especially diversity; would have loved to try DAE.
4. Postprocessing, but no time and unstable CV.
5. hparam tuning in task 2 - we really ran out of time near the end (Stanley's alarm didn't work) so our task 2 is very likely, bad. Really bad. Wouldn't be surprised if we lost to AutoML.
6. SIR model (couldn't quite tune parameters)
7. Finding a nicer preprocessing, perhaps with anomaly detection/seasonality
If you would like to see our ideation process, check out `ideas.md` - it features our knowledge from other applications that we would have liked to try. 
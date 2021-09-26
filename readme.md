# DCP COVID-19 Challenge

24 hours, 1700 lines of code, and 74 models, and here we are. This was so much fun, thanks to the organizers.
### Our solution
- Stack models (in total, 74 models)
    - Diverse ensemble consisting of many types of models trained with different schemes (see Schemes below).
- Stack after each single prediction so that following predictions are made on more accurate data. We stack with a weighted average based on val MSE
- Robust cross validation. For cross validation, we use 3 folds, one for each CSV. We scale each CSV individually so that this is possible.
- Feature engineering for some models. Using meta labels, such as cumulative case counts, total vaccinated, etc. we get much better MAE, faster convergence, and our model does not "lag" one day behind in our CV; this indicates better robustness.
- A lot more to talk about but no time. We will update this readme on GitHub to provide our full solution.

### All 74 models
| path                                           | mse  | graph overlap? | type |
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

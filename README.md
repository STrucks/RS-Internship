# RS-Internship

## ZSL with hyper-spectral data:

### Data distribution:

| \            | % of whole dataset |
|:-----------: |:------------------:|
|      1       | 0.00448824275539077 |
|      2       | 0.13933066640647868  |
|      3       | 0.08098351058639867 |
|      4       | 0.023124207239730705|
|      5       | 0.047126548931603084|
|      6       | 0.07122646111815786|
|      7       | 0.002731973851107425|
|      8       | 0.04663869645819104|
|      9       | 0.0019514098936481608|
|     10       | 0.09483852083130062|
|     11       | 0.23953556444531174|
|     12       | 0.057859303346667966|
|     13       | 0.020001951409893647|
|     14       | 0.12342667577324618|
|     15       | 0.03766221094740951|
|     16       | 0.009074056005463947|


## Data Preprocessing:
Data normalization works. Balancing also works.

Data distribution:

![Data Distribution](https://github.com/STrucks/RS-Internship/blob/master/exploration%20imgs/balanced_dataset.png)

To make the data really balanced, we only take the classes with 100 or more samples.

Lets see how well the autoencoder encodes and decodes:

![original](https://github.com/STrucks/RS-Internship/blob/master/exploration%20imgs/original_example.png)

![reconstruction](https://github.com/STrucks/RS-Internship/blob/master/exploration%20imgs/reconstruction_example.png)

## Losses:
Typical loss of the classifier:

![ZSL classif with PCA features losss](https://github.com/STrucks/RS-Internship/blob/master/exploration%20imgs/loss_with_pca_features.png)


### Experiments:
| Approach           | left-out class | mean accuracy on test set | std of accuracy | mean accuracy on left-out classes
|:-----------: |:------------------:|:------------------:|:------------------:|:------------------:|
|     idea 1                                     | - | 30.5% | 2.45% | - |
|     idea 1                                     | 2 | 28.9% | 1.53% | 0.0%|
|     idea 1                                     | 2,3 | 30.2% | 1.97% | 0.0%|
|     idea 1                                     | 2,3,4 | 30.2% | 1.97% | 0.0%|
|     idea 1                                     | 2,3,4,5 | 37.3% | 2.66% | 0.0%|
|     idea 2                                     | - | 34% | 1.37% | - |
|     idea 2                                     | 2 | 29.7% | 4.8% | 0.0%|
|     idea 2                                     | 2,3 | 34% | 1,37% | 0.0%|
|     idea 2                                     | 2,3,4 | 32.4% | 1.37% | 0.0%|
|     idea 2                                     | 2,3,4,5 | 36.4% | 2.91% | 0.0%|
|     idea 2                                     | 2,3,4,5 | 36.4% | 2.91% | 0.0%|
|     auto encoder features                      | - | 39.6% | 4.14% | 0.0%|
|     auto encoder features                      | 2 | 27.8% | 1.3% | 0.0%|
|     auto encoder features                      | 2,3 | 36.7% | 3.31% | 0.0%|
|     auto encoder features                      | 2,3,4 | -% | -% | 0.0%|
|     auto encoder features                      | 2,3,4,5 | -% | -% | 0.0%|
|     variantional auto encoder features         | - | 30.9% | 1.53% | 0.0%|
|     variantional auto encoder features         | 2 | 32.5% | 1.4% | 0.0%|
|     variantional auto encoder features         | 2,3 | 35.1% | 4.9% | 0.0%|
|     variantional auto encoder features         | 2,3,4 | 28.9% | 3.1% | 0.0%|
|     variantional auto encoder features         | 2,3,4,5 | 29.7% | 2.7% | 0.0%|











# RS-Internship

## ZSL with hyper-spectral data:

### First results:
The first results obtained involve an empty attribute matrix. 
Parameters:
* learning rate = 0.00001
* size of the hidden layer of attribute network = 1000
* size of the hidden layer of relation network = 1000
* epochs = 400

This set up was tested with varying numbers of left out classes (zero shot classes):
* no classes left out: accuracy score of ~14% 
* class number [16] left out: accuracy score of ~13,4%
* class number [15,16] left out: accuracy score of ~8%

The accuracy drops because of a) the nature of the ZSL problem and b) because the attribute matrix is still 0.


### More results:
After implementing feature extraction with various methods (see code), I came to more results:

* learning rate = 0.0001
* size of the hidden layer of attribute network = 100
* size of the hidden layer of relation network = 100
* epochs = 20

This set up was tested with varying numbers of left out classes (zero shot classes). See table:


| \            | simple features | abst features 1 | abst features 2 | abst features 3 |
|:-----------: |:---------------:|:---------------:|:---------------:|:---------------:|
| all classes  |               	 |       80%, 0.10     	 |      16%, 0.08      	 |       13%, 0.08   	 |
| no [16]      |               	 |       84%, 0.09     	 |      3%       	 |       5%      	 |
| no [15,16]   |               	 |       80%, 0.12     	 |      5%       	 |       4%      	 |
| no [11-16]   |               	 |       50%, 0.15     	 |      5%       	 |       4%      	 |


I have the impression that the optimizer gets stuck in a local optima, mostly with an accuracy of ~6%, ~0.4% (baseline).
It is striking that the model archives the same result most of the time. My guess is that the optimizer gets stuck in a local optimum that is extremely bad (propably predicting only one class). There are some occasions where the accuracy on the test set is high (upto 40%!!). In these cases, the optimizer finds a better solution and performs reasonable. Thats why I conducted the experiments over fifteen trials and only reported the average accuracy.
The accuracy for one individual trial was often either 0.004697040864255519 or 0.13433536871770785. This might be the baseline of one class, because we have an unbalanced dataset. Probably the baseline of class 2 and class 1. However, I dont know why the classifier would choose class 2 as the best class to guess for.

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




### Further steps:
Maybe change optimizer? other than Adam

One should mention that the accuracy scores are not cross validated.


---------------------------------------------------

## Data Preprocessing:
Data normalization works, data augmentation however does not work (see distributions below).

![Data Distribution](https://github.com/STrucks/RS-Internship/blob/master/exploration%20imgs/data%20distr.png)

## Reducing the Autoencoder Network:
That does not make it better or worse. The features are slightly different after 50 epochs of training. That is however an improvement towards the features with a deeper neural network, which were exactly the same for every class. A heatmap of the features can be seen below. 

![Heatmap](https://github.com/STrucks/RS-Internship/blob/master/exploration%20imgs/autoencoder%20features.png)

You can see the loss after every epoch drop and converge:

![Loss](https://github.com/STrucks/RS-Internship/blob/master/exploration%20imgs/loss%20for%20autoencoder.png)

The performance of the ZSL classifier however stays the same (~ 24% accuracy). It still always returns the label of the most probable class. 

Features of VAE:

![VAE features](https://github.com/STrucks/RS-Internship/blob/master/exploration%20imgs/VAE_features.png)

Confusion matrix: 

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0. 270.   0.  88. 207.   0.   0.   0. 105. 270.   0. 205. 904.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]]

The loss in the end of the training phase is 0.0753, the accuracy on the training set is about 27%.

Also, it seems like the load_data function from utils does not remove the 0 class.

------------------------------------------------------------------------------------
### After balancing the data set:
Data distribution:

![Data Distribution](https://github.com/STrucks/RS-Internship/blob/master/exploration%20imgs/balanced_dataset.png)

To make the data really balanced, we only take the classes with 100 or more samples.

Lets see how well the autoencoder encodes and decodes:

![original](https://github.com/STrucks/RS-Internship/blob/master/exploration%20imgs/original_example.png)

![reconstruction](https://github.com/STrucks/RS-Internship/blob/master/exploration%20imgs/reconstruction_example.png)

## Losses:
The loss of the ZSL classifier does not decrease even after 50 epochs. 

To see if the classifier actually works, I crafted a feature matrix by hand (Identity matrix). If we use that as feature matrix, the classifier actually learns:

![loss](https://github.com/STrucks/RS-Internship/blob/master/exploration%20imgs/loss_with_perfect_features.png)

confusion matrix (100 epochs):
[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[346.  83.   0.   0.   0.  28. 938. 282.   0.   0.   0.   0.   0.   0.    0. 905.]

[118.  40. 194.   0. 376. 178. 425.  65. 183.  12.   0.   0.   0.   0.    0. 224.]

[  0.  14.  48. 356.   2.   7.  38.  23.   0. 144.   0.   0.   0.   0.    0.  14.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[261.   0.   0.   0.   0. 565. 636.  70.   0.   0.   0.   0.   0.   0.    0. 130.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  5.   0.   0.   0.   0.  77. 296.  16.   0.   0.   0.   0.   0.   0.    0.  36.]

[  0.   0. 141. 274.   0.   0.   3.   0. 982. 130.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.    0.   0.]

[  0.   0.   0.   0.   0.  17.  19.  37.   0.   0.   0.   0.   0.   0.    0.  19.]

Accuracy of 25.3% on test set.

We reach an accuracy of about 20 to 25%, which is better than chance (1/11). 
Learning rate: 0.001, no weight decay, augmentation type = n.

My conclusion is that the classifier should be able to learn the differences, but since the differences are extremely small, it needs to learn either much longer or use a better attribute matrix.


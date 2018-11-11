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

* learning rate = 0.000001
* size of the hidden layer of attribute network = 1000
* size of the hidden layer of relation network = 1000
* epochs = 500

This set up was tested with varying numbers of left out classes (zero shot classes). See table:


| \            | simple features | abst features 1 | abst features 2 | abst features 3 |
|:-----------: |:---------------:|:---------------:|:---------------:|:---------------:|
| all classes  |               	 |       7%     	 |      4%      	 |       6%      	 |
| no [16]      |               	 |       7%      	 |      3%       	 |       5%      	 |
| no [15,16]   |               	 |       6%      	 |      5%       	 |       4%      	 |


I have the impression that the optimizer gets stuck in a local optima, mostly with an accuracy of ~6%, ~0.4% (baseline).
It is striking that the model archives the same result most of the time. My guess is that the optimizer gets stuck in a local optimum that is extremely bad (propably predicting only one class). There are some occasions where the accuracy on the test set is high (upto 40%!!). In these cases, the optimizer finds a better solution and performs reasonable. Thats why I conducted the experiments over fifteen trials and only reported the average accuracy.
The accuracy for one individual trial was often either 0.004697040864255519 or 0.13433536871770785. This might be the baseline of one class, because we have an unbalanced dataset. However, these baselines do not match any baselines (except maybe class 16), see tabel below

| \            | % of whole dataset |
|:-----------: |:------------------:|
|      0       | 0.5125326991676575 |
|      1       | 0.0021878715814506538 |
|      2       | 0.0679191438763377  |
|      3       | 0.03947681331747919 |
|      4       | 0.01127229488703924|
|      5       | 0.02297265160523187|
|      6       | 0.034720570749108205|
|      7       | 0.0013317479191438763|
|      8       | 0.022734839476813318|
|      9       | 0.0009512485136741974|
|     10       | 0.04623067776456599|
|     11       | 0.11676575505350772|
|     12       | 0.028204518430439952|
|     13       | 0.009750297265160524|
|     14       | 0.06016646848989298|
|     15       | 0.01835909631391201|
|     16       | 0.004423305588585018|




### Further steps:
Maybe change optimizer? other than Adam

One should mention that the accuracy scores are not cross validated.


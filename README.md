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

This set up was tested with varying numbers of left out classes (zero shot classes):
* no classes left out: 
** accuracy score of up to 40% with abstract features 1
** accuracy score of up to 40% with abstract features 1
** accuracy score of up to 40% with abstract features 1
** accuracy score of up to 40% with abstract features 1

* class number [16] left out: 
** accuracy score of up to 40% with abstract features 1
** accuracy score of up to 40% with abstract features 1
** accuracy score of up to 40% with abstract features 1
** accuracy score of up to 40% with abstract features 1

* class number [15,16] left out: accuracy score of ~8%

| \            | simple features | abst features 1 | abst features 2 | abst features 3 |
|:-----------: |:---------------:|:---------------:|:---------------:|:---------------:|
| all classes  |               	 |       40%     	 |      14%      	 |               	 |
| no [16]      |               	 |               	 |               	 |               	 |
| no [15,16]   |               	 |               	 |               	 |               	 |


I have the impression that the optimizer gets stuck in a local optima, mostly with an accuracy of ~6% (baseline).


### Further steps:
One should mention that the accuracy scores are not cross validated.
Investigate characteristic/typical features for hyp-spect-data

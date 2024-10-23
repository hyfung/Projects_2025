# Random Forest

## Principle

### Bootstrapping
- Extract a subset from original data to form new dataset

### Random Feature Selection
- Extract a random attribute from subset to use as a split

### Entropy
![](./images/entropy.png)
- Uncertainty in a dataset
- − i=1∑n ​pi​log2​(pi​)
- 0 -> Single class, no uncertainty
- 1 -> Equal distribution of class, max certainty
- Random forest creates classifier to reduce entropy

### Information Gain
![](./images/information_gain.png)
- Reduction of entropy after a split by attribute
- Quantifies how well an attribute separates a class

## Parameters

## Training

## Classification

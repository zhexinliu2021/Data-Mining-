#  DATASET AND PROBLEM

As a first step, let us look at the dataset we are faced with. The domain from which the dataset originates is the domain of mental health. More and more smartphone applications are be- coming available to support people suffering a depression. These applications record all kinds of sensory data about the behavior of the user and in addition frequently ask the user for a rating of the mood. A snapshot of the resulting dataset is shown in Table 1. The dataset contains ID’s, reflecting the user the measurement originated from. Furthermore, it contains time-stamped pairs of variables and values. The variables and their interpretation are shown in Table 2.
Using this dataset, we would like to build a predictive model that is able to predict the average mood of the user on the next day based on the data we obtained from the user on the days before. This is illustrated graphically in Figure 1 below. In order to create such a predictive model, we need to perform some transformations and we need to decide on what features we want to use for these predictions.

   

# Epilepsy Diagnosis Using EEG Signals

Diagnosis system that predicts whether a person has epilepsy or not from an input EEG signal, by applying different machine learning algorithms.


# Dataset (Original):

- A real-world data, consists of 500 patients.
- Patients are equally divided into 5 sets (Z, O, F, N, S) (i.e. 100 patients each) : 
		- **Z** for healthy people in relaxed and awake state.
		- **O** for healthy people with eye movements.
		- **N and F** for interictal people. 
		- **S** for ictal people.
- Each file contains 4097 values, in .txt file, which represents the brain activity for 23.6 secs.
- **Z and O** sets represent **normal** people. While **N, F and S** represent **up normal** people.



## Classification Algorithms :

- Bayes classifier with Maximum-Likelihood-Estimation (MLE).
- K-Nearest Neighbor (KNN).
- Multi-Layer Perceptron Neural Network. 


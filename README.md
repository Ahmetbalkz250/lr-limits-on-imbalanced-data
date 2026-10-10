# Limits of Logistic Regression on Imbalanced Data

In this project, I attempted to train a logistic regression model on a highly imbalanced dataset using two common techniques: `class_weight` and `SMOTE`. Despite applying these methods, the model failed to achieve satisfactory performance. 

As visualized in the analysis below, the core issue is not just the class imbalance, but the non-linear, overlapping nature of the data points. This geometric overlap makes linear models like Logistic Regression fundamentally inadequate for separating the classes in this specific problem.

# Quora-question-pairs
Models used for the Kaggle competition: https://www.kaggle.com/c/quora-question-pairs

Approach consisted in XGBoost model and Siamese LSTMs. The Siamese network was used to compute the distance between questions and its output of was used as features in the XGBoost model, together with some other features like word match of the questions, character length, etc.

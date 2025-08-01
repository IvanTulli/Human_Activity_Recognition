# Human_Activity_Recognition
We use the "Human Activity Recognition with Smartphones" dataset from https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones. 

We try to find a good model for predicting the activity a person is doing (among standing, sitting, laying, walking, walking downstairs, and walking upstairs), based on the readings of the gyroscope and accelerometer of their smarthphone.

As a baseline we first train a logistic regression model and measure the predicted accuracy and F1 scores on the test set. We then compare the logistic regression model with two neural network models. One of them is trained on the original dataset, while the other is trained on a dimensionally reduced space of features, obtained from the encoder part of
an autoencoder network. We find that the simple logistic regression beats both neural network models in both accuracy and F1 scores. 

The uploaded files are:
- The Jupyter notebook.
- The dataset split into train and test sets.
- The best keras models obtained during training (in terms of validation loss). These include the two neural network models and the autoencoder. 

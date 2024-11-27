Intruduction:

This research centers on Bacillus subtilis, a model organism of Gram-positive bacteria known for its broad industrial applications, aiming to predict the efficiency of signal peptide secretion. By employing automatic machine learning technology (AutoML) and integrating data from a high-throughput library of signal peptides, we have developed a predictive model for the secretion efficiency of signal peptides in Bacillus subtilis. The experimental outcomes reveal that our regression model achieves a coefficient of determination (R²) of 0.3636, while the area under the receiver operating characteristic curve (AUROC) for the classification model stands at an impressive 91.11%. The trained classification model SecEff-Pred has been successfully deployed as a web server, accessible to users through the website http://www.lamee.cn/web_service.

How to use it ?

1. If you want to train a classification model, please run the classification/code/classification.py. After you get the trained model, you can run the confusion_matrix_test.py to obtain the result of the calculation of the confusion matrix on the test set or run the t-SNE.py to visualize the extracted embedding vectors.
2. If you want to train a regression model, please run the regression/code/regression.py.
3. You can download the pre-trained model from the URL https://huggingface.co/google/electra-base-discriminator and save it to the model_checkpoint folder locally.

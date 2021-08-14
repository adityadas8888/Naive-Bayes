# Naive Bayes
<h4><i>This program was part of an assignment for CSE 5334 Data Mining at University of Texas at Arlington</i></h4>
( Also if you are a student taking CSE 5334 at UTA, and you are blatantly copying the code, kindly know that you might be caught for plagiarism. Copy at your own risk)
 <br>
<br>
<b>We are assuming that the labels remain to 0 and 1, and the training and testing datasets are created in
the file and not provided by the user.</b>
<br>
<u><b>Functions created:</b></u>
<br>
<u><b>naïve_bayes(trainingset0_size,trainingset1_size)</b></u>
trainingset0_size,trainingset1_size are the sizes of the training set of label 0 and label 1 respectively.
This function is called in the main function and the sizes of the training sets can be adjusted.
<br>
<u><b>find_roc(tpr,fpr)</u></b>
<br>
Tpr and fpr are columns of a pandas dataframe that are passed to the function to calculate and draw the
roc curve. The function call is commented by default under the naïve_ bayes function and needs to be
uncommented to get roc graphs.
<br>
<b><u>Area_curve(maxx, maxxy)</b></u>
<br>
Maxx and maxy are the max value of tpr and fpr respectively. They are required to calculate the area
under the curve. We are approximating the area by taking the area of a trapezoid. This function call is
commented by default under find_roc and needs to be uncommented to get the AUC
<br>
<h4><b><u>Dataframe results</h4></b></u>
<br>The following testing dataframe is generated(but not printed) upon runnng of the program.
<img src="https://github.com/adityadas8888/Naive-Bayes/blob/master/images/im1.png" width="550"/>
<img src="https://github.com/adityadas8888/Naive-Bayes/blob/master/images/im2.png" width="550"/>
<br>
Posterior probability and pred are generated in the dataframe. The error is
calculated after the generation of the dataframe.
<img src="https://github.com/adityadas8888/Naive-Bayes/blob/master/images/im3.png" width="550"/>
<br>
Scatter plot, accuracy and confusion matrix when training set is 500,500.
Red points are 0, blue points are 1.
<br>
<img src="https://github.com/adityadas8888/Naive-Bayes/blob/master/images/im4.png" width="550"/>
<br>
Scatter plot, accuracy and confusion matrix when training set is 10,10.
Red points are 0, blue points are 1.
<br>
<img src="https://github.com/adityadas8888/Naive-Bayes/blob/master/images/im5.png" width="550"/>
<br>
Scatter plot, accuracy and confusion matrix when training set is 20,20.
Red points are 0, blue points are 1.
<img src="https://github.com/adityadas8888/Naive-Bayes/blob/master/images/im6.png" width="550"/>
<br>
Scatter plot, accuracy and confusion matrix when training set is 50,50.
Red points are 0, blue points are 1.
<img src="https://github.com/adityadas8888/Naive-Bayes/blob/master/images/im7.png" width="550"/>
<br>
Scatter plot, accuracy and confusion matrix when training set is 100,100.
Red points are 0, blue points are 1.
<img src="https://github.com/adityadas8888/Naive-Bayes/blob/master/images/im8.png" width="550"/>
<br>
Scatter plot, accuracy and confusion matrix when training set is 300,300.
Red points are 0, blue points are 1.
<img src="https://github.com/adityadas8888/Naive-Bayes/blob/master/images/im9.png" width="550"/>
<br>
Scatter plot, accuracy and confusion matrix when training set is 500,500.
Red points are 0, blue points are 1.
<img src="https://github.com/adityadas8888/Naive-Bayes/blob/master/images/im10.png" width="550"/>
<br>
Plot of the accuracies.
The accuracy for the most part tends to increase with the increasing size of the
dataset. This can be contributed to the part that we would be getting gaussian
curve and likelihood with a higher training dataset.
<br>
<img src="https://github.com/adityadas8888/Naive-Bayes/blob/master/images/im11.png" width="550"/>
<br>
Scatter plot, accuracy and confusion matrix when training set is 300,700.
Red points are 0, blue points are 1.
The accuracy decreases, in comparison to the 500,500 dataset. This might be
because, the training model was skewed towards the label 1. Hence, we end up
with a lower accuracy. Most of the labels were erroneously labelled 1.
<br>
<img src="https://github.com/adityadas8888/Naive-Bayes/blob/master/images/im12.png" width="550"/>
<br>
ROC for 500,500 dataset
<br>
<img src="https://github.com/adityadas8888/Naive-Bayes/blob/master/images/im13.png" width="550"/>
<br>
<img src="https://github.com/adityadas8888/Naive-Bayes/blob/master/images/im14.png" width="550"/>
<br>
ROC for 300, 700 dataset
<br>
<img src="https://github.com/adityadas8888/Naive-Bayes/blob/master/images/im15.png" width="550"/>
<br>

This code folder contains python code for research article: "A New Look at Breathing for Affective Studies" submitted to "IEEE TRANSACTIONS ON AFFECTIVE COMPUTING"
If you only want to run all experiments in one shot, make sure you checkout both 'data' and 'code' folders and their files.
Then you can run 'runAllScripts.py', it will read in the raw data file 'data\BR_Filtered_Raw.csv', and generate the following files:
1. featuresNew_12_cross.csv
2. Figure 9, 10, 11a, 11b, 11c in the article.
3. A folder 'waveformImages' that stores all waveforms from the experiments. (figure 4,5,6 in the article)

If you want to explore our code, please read the following section carefully.
We have 3 python files in code folder. They are used to preprocess the data, generate the cross-correlation matrix, and train and validate the multinominal model
These 3 files have dependencies on each other and need to be run in following order.
First run WaveformProcess.py, it will generate a csv file 'featuresNew_12_cross.csv'
Second run CorrelationMatrix.py, it will generate a Figure8.png, for user to analysis the correlation matrix of 18 features.
Third run LogisticRegression_MulNominal_one.py, it will build a Mulinominal model and cross-validate that model.

In the following section, we explain in details how each python code work:
1. WaveformProcess.py: this code will do 3 things, 
1.1 Load the raw data set from '..\\data\\BR_Filtered_Raw.csv' and apply smoothing and normalization filter to it to retrieve the S' signal and normalized S" signals
1.2 Use the S" signals to Identify each breathing cycle(wave)
1.3 Calculate all 18 statistic features of all cycles per participant and treatment
The final result will be saved to a csv file 'featuresNew_12_cross.csv'
2. The CorrelationMatrix.py will load 'featuresNew_12_cross.csv' and draw the cross-correlations between these 18 features. The figure is named as 'Figure9.png'. Based on this map, we created table 2, and carefully select features to tune the model.
3. The LogisticRegression_MulNominal_one.py will do following things:
3.1 Load and clean the 'featuresNew_12_cross.csv' dataset. We removed all na records,records that its participant do not have 'RB' treatments,  and all RB records. Only 'ST', 'LT', 'PR' records are kept for model training and validation.
3.2 Then we build a one vs. rest multinominal model and generate model report in Figure 10, and table 3 in the article.
3.3 Last, we use a 3-fold cross validation process to validate our methods. The results of 3-fold cross validation are given in Figure11a,b,c.


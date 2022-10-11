# Building the Logistic Regression Model:

# Importing libraries:
import pandas as pd
import numpy as np
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import count, isnan, udf, when
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import pyspark.sql.functions as F
from pyspark.sql.functions import col, isnan, when, count
from pyspark.sql.functions import avg
from pyspark.sql.types import StringType, DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import Imputer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import time
start_time = time.time()

"""Setting a Spark Session:"""

# Building Spark Session:
spark = SparkSession.builder.master("yarn").config('spark.executor.cores', 4).appName("LogReg").getOrCreate()

# Reading in the data: 
df = spark.read \
    .option("inferSchema",True) \
    .option("sep",",") \
    .option("header",True)\
    .option("treatEmptyValuesAsNulls","true")\
    .option("nullValue", "NA")\
    .csv("Dataset.csv")

# Row and column count of df:
print('A total of', df.count(), 'rows\n')
print('A total of', len(df.columns), 'columns\n')

# Printing the first 10 rows of the df:
df.show(10)

# Printing df Schema
df.printSchema()

# Print all columns presents as a list:
df.columns

# Selecting a random sample for EDA - Exploratory Data Analysis:
EXdf = df.sample(0.06, 20220804) # Selecting 6% and assigning a seed to ensure same sample is picked each time.

# Counting the row number of our sample.
EXdf.count()

# Exploring the sample:
EXdf.describe().show()

# Explore the target variable:
EXdf.describe(['hospital_death']).show()

# Seeing the distribution of the target variable in the sample:
EXdf.groupBy('hospital_death').count().show()
print("We can clearly see an imbalance is present in the target varibale.",
      "Which will need to be addressed when prepping the model \n")

# Adding a percentage view of the target variable distribution.
tot = EXdf.count()

EXdf.groupBy('hospital_death') \
  .count() \
  .withColumnRenamed('count', 'cnt_per_group') \
  .withColumn('perc_of_count_total', (F.col('cnt_per_group') / tot) * 100 ) \
  .show()

# Converting our exploratory sample into a pandas dataframe so we can visualise a few aspects:
pandasDF = EXdf.toPandas()
print(pandasDF)

# Visualising the distribution of the target variable:
sns.set_style("darkgrid")
target = sns.countplot(x='hospital_death', data=pandasDF)
plt.title('Target Variable - "hospital_death" Distribution')
plt.show()

# Visualising the distribution of the target variable by gender:
sns.set_style("darkgrid")
gender = sns.countplot(x='hospital_death', data=pandasDF, hue='gender', palette="magma")
plt.title('Hospital Deaths by Gender')
plt.show()

# Visualising the distribution of deaths between hospital dept.:
sns.set_style("darkgrid")
sns.set(rc={'figure.figsize': (22, 10)})
gender = sns.countplot(x='hospital_death', data=pandasDF, hue='hospital_admit_source', palette="hls")
plt.title('Hospital Deaths by Department')
plt.show()

# Visualising the distribution of the deaths between medical category:
sns.set_style("darkgrid")
sns.set(rc={'figure.figsize': (22, 10)})
gender = sns.countplot(x='hospital_death', data=pandasDF, hue='apache_2_bodysystem', palette="rocket")
plt.title('Hospital Deaths by Medical Aspect ')
plt.show()

# Visualing the degree of null values present within the sample:
msno.bar(pandasDF)
print('\n')
msno.matrix(pandasDF)
plt.show()

# Importing the data dictionary to better understand the target variable:
# Reading in the data: 
dict_df = spark.read \
    .option("inferSchema",True) \
    .option("sep",",") \
    .option("header",True)\
    .csv("DataDictionary.csv")

dict_df.show(10)
print('Based on the preview of the data-frame we can see multiple binary type columns present in the core dataset. \n')

# Remove spaces from column names in the data dictionary.
renamed_dict_df = dict_df.select([F.col(col).alias(col.replace(' ', '_')) for col in dict_df.columns])

# Checking the replacement worked.
renamed_dict_df.show()

# Idenitifying other binary data types as per the data dictionary.
renamed_dict_df.filter(renamed_dict_df.Data_Type == "binary").show(truncate=False)

"""Cleaning the dataset:"""

# Evaluating count of null.
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
   ).show()

# Dropping columns which represent ID's as they play no signifance to the overall model.
condition = lambda col: '_id' in col
df = df.drop(*filter(condition, df.columns))
print('A total of', len(df.columns), 'columns is left after dropping ID columns\n')

# Dropping columns with a Null values of the threshold of 7500.

# First we aggregate the dataframe by using the collect action to collect the count of null values in each column.
# This return as an array.
aggregated_row = df.select([(count(when(col(c).isNull(), c))/df.count()).alias(c) for c in df.columns]).collect()
print(list(aggregated_row))

# Converting this array to a dictionary.
aggregated_dict_list = [row.asDict() for row in aggregated_row]
aggregated_dict = aggregated_dict_list[0]

# Using dictionary apprehensions to find columns in which the null count is greater than 10% of the overall total.
col_null_g_10p=list({i for i in aggregated_dict if aggregated_dict[i] > 0.10})

# Printing all columns fitting this criteria.
print(col_null_g_10p, '\n' )
  
# Dropping these columns.
df = df.drop(*col_null_g_10p)

# Checking how many columns left. 
print(len(df.columns))

# Based on the data from the data dictionary, we drop all binary data types except for 'hospital_death' as this is the target variable.
df = df.drop('elective_surgery ',
            'readmission_status',
            'apache_post_operative',
            'arf_apache',
            'gcs_unable_apache',
            'intubated_apache ',
            'ventilated_apache',
            'aids ',
            'cirrhosis',
            'diabetes_mellitus',
            'hepatic_failure  ',
            'immunosuppression',
            'leukemia',
            'lymphoma',
            'solid_tumor_with_metastasis')

# Checking how many columns left. 
print(len(df.columns))

# Commenting as Schema applies same logic.
# Identifying categorical type columns in the dataframe:
# str_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
# print(str_cols)
# print(len(str_cols))

# Dropping null values from above columns, as the number of null values is not impactfull to overall count.
# Count before dropping rows:
print('A total of', df.count(), 'rows\n')

df = df.dropna(subset=('ethnicity', 'gender', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem'))

# Count after dropping rows:
print('A total of', df.count(), 'rows\n')

# Viewing categorical data present in each category column.
df.select('ethnicity').distinct().show()
df.select('gender').distinct().show()
df.select('icu_admit_source').distinct().show()
df.select('icu_stay_type').distinct().show()
df.select('apache_3j_bodysystem').distinct().show()
df.select('apache_2_bodysystem').distinct().show()

# Converting all string type or categorical type coplumns/data present within the dataset to an index category in prep for vectorAssembler.

# Creating a list of  columns that are string typed
categoricalColumns = [item[0] for item in df.dtypes if item[1].startswith('string') ]

# Defining a list of stages in the pipeline. The string indexer will be one stage.
stages = []

# Iterating through all categorical values:
for categoricalCol in categoricalColumns:
    # Creating a string indexer for those categorical values and assign a new name including the word 'Index'
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')

    # Appending the string Indexer to the list of stages
    stages += [stringIndexer]

# Creating the pipeline. Assigning the stages list to the pipeline key word stages
pipeline = Pipeline(stages = stages)

# Fitting the pipeline to our dataframe
pipelineModel = pipeline.fit(df)

# Transforming the dataframe
df= pipelineModel.transform(df)

# Now we drop the original columns:
cols = ('ethnicity', 'gender', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem')
df = df.drop(*cols)

# Re-evaluating count of null.
# df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
#    ).show()

# Dropping null values from 'age', 'bmi', 'height', 'weight' as they wouldn't fit the criteria of filling null values using mean of column.
# Count before dropping rows:
print('A total of', df.count(), 'rows\n')

df = df.dropna(subset=('age', 'bmi', 'height', 'weight'))

# Count after dropping rows:
print('A total of', df.count(), 'rows\n')

print('Overall the reduction in the dataset is not extreme and we have a large data sample to work with further.')

# Commenting as Schema applies same logic.
# Identifying integer type columns in the dataframe:
int_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, IntegerType)]
print(int_cols)
print(len(int_cols))

# Commenting as Schema applies same logic.
# Identifying double type columns in the dataframe:
dbl_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, DoubleType)]
print(dbl_cols)
print(len(dbl_cols))

# Now we move to addressing the remaining null values present in our numeric columns using the mean of the columns in which they are present.
# First we build a function to find the avg or mean of a column:
def mean_of_pyspark_columns(df, numeric_cols, verbose=False):
    col_with_mean=[]
    for col in numeric_cols:
        mean_value = df.select(avg(df[col]))
        avg_col = mean_value.columns[0]
        res = mean_value.rdd.map(lambda row : row[avg_col]).collect()
        
        if (verbose==True): print(mean_value.columns[0], "\t", res[0])
        col_with_mean.append([col, res[0]])    
    return col_with_mean

# Fill missing values for mean
from pyspark.sql.functions import when, lit

def fill_missing_with_mean(df, numeric_cols):
    col_with_mean = mean_of_pyspark_columns(df, numeric_cols) 
    
    for col, mean in col_with_mean:
        df = df.withColumn(col, when(df[col].isNull()==True, 
        lit(mean)).otherwise(df[col]))
        
    return df

# Setting a list iwth numeric columns, then applying the fill with missing mean function to the dataframe.
numeric_cols= ['age',
               'elective_surgery',
               'apache_2_diagnosis',
               'gcs_eyes_apache',
               'gcs_motor_apache',
               'gcs_verbal_apache', 
               'heart_rate_apache',
               'intubated_apache',
               'map_apache',
               'd1_diasbp_max',
               'd1_diasbp_min',
               'd1_diasbp_noninvasive_max',
               'd1_diasbp_noninvasive_min',
               'd1_heartrate_max',
               'd1_heartrate_min',
               'd1_mbp_max',
               'd1_mbp_min', 
               'd1_mbp_noninvasive_max',
               'd1_mbp_noninvasive_min',
               'd1_resprate_max',
               'd1_resprate_min',
               'd1_spo2_max', 
               'd1_spo2_min', 
               'd1_sysbp_max',
               'd1_sysbp_min', 
               'd1_sysbp_noninvasive_max', 
               'h1_diasbp_max',
               'h1_diasbp_min', 
               'h1_diasbp_noninvasive_max', 
               'h1_diasbp_noninvasive_min',
               'h1_heartrate_max', 
               'h1_heartrate_min',
               'h1_mbp_max',
               'h1_mbp_min', 
               'h1_mbp_noninvasive_max',
               'h1_mbp_noninvasive_min',
               'h1_resprate_max', 
               'h1_resprate_min', 
               'h1_spo2_max', 
               'h1_spo2_min', 
               'h1_sysbp_max', 
               'h1_sysbp_min', 
               'h1_sysbp_noninvasive_max',
               'h1_sysbp_noninvasive_min', 
               'd1_glucose_max', 
               'd1_glucose_min', 
               'aids', 
               'hepatic_failure', 
               'bmi', 
               'height', 
               'pre_icu_los_days', 
               'weight', 
               'apache_3j_diagnosis', 
               'resprate_apache', 
               'temp_apache',
               'd1_sysbp_noninvasive_min',
               'd1_temp_max', 
               'd1_temp_min', 
               'apache_4a_hospital_death_prob',
               'apache_4a_icu_death_prob']

df = fill_missing_with_mean(df, numeric_cols)

# Confirming all null values have been dealt with.
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
   ).show()

# Lets review the distribution in the whole dataset:
df.groupBy('hospital_death').count().show()

# Calculating ratio
major_df = df.filter(col("hospital_death") ==0)
minor_df = df.filter(col("hospital_death") == 1)
ratio = int(major_df.count()/minor_df.count())
print(ratio)
print("ratio: {}".format(ratio))

print('We will address this imbalance by undersmapling the majority class since we have such a large dataset to work with.')

# Undersampling the majority class.
sampled_majority_df = major_df.sample(False, 1.5/ratio)
df = sampled_majority_df.unionAll(minor_df)
df.show()

# Lets review if the distribution is less imbalanced now:
df.groupBy('hospital_death').count().show()

# Converting our target variable to String type data prior to encoding.
df.select(col("hospital_death").cast('string').alias("hospital_death"))

# Moving to encoding the target variable as 'True' / 'False'
binary_encode = udf(lambda x: "False" if x==0 else "True", StringType())

df = df.withColumn("Death_Outcome", binary_encode("hospital_death")).drop("hospital_death")
df.show(10)

print('Having undersampled our selection to allow for a more even distribution of True/False, we currently have a total of:', len(df.columns), 'columns and a total of:', df.count(), 'rows \n')

# Now lets print all columns name's before we convert the Spark Dataframe into the sparse format expected by the Machine Learning.
# We have addressed all null values in preperation to this.
df.columns

# We only covert our features first:
cols = ['age',
        'bmi',
        'elective_surgery',
        'height',
        'pre_icu_los_days',
        'weight',
        'apache_2_diagnosis',
        'apache_3j_diagnosis',
        'gcs_eyes_apache',
        'gcs_motor_apache',
        'gcs_verbal_apache',
        'heart_rate_apache',
        'intubated_apache',
        'map_apache',
        'resprate_apache',
        'temp_apache',
        'd1_diasbp_max',
        'd1_diasbp_min',
        'd1_diasbp_noninvasive_max',
        'd1_diasbp_noninvasive_min',
        'd1_heartrate_max',
        'd1_heartrate_min',
        'd1_mbp_max',
        'd1_mbp_min',
        'd1_mbp_noninvasive_max',
        'd1_mbp_noninvasive_min',
        'd1_resprate_max',
        'd1_resprate_min',
        'd1_spo2_max',
        'd1_spo2_min',
        'd1_sysbp_max',
        'd1_sysbp_min',
        'd1_sysbp_noninvasive_max',
        'd1_sysbp_noninvasive_min',
        'd1_temp_max',
        'd1_temp_min',
        'h1_diasbp_max',
        'h1_diasbp_min',
        'h1_diasbp_noninvasive_max',
        'h1_diasbp_noninvasive_min',
        'h1_heartrate_max',
        'h1_heartrate_min',
        'h1_mbp_max',
        'h1_mbp_min',
        'h1_mbp_noninvasive_max',
        'h1_mbp_noninvasive_min',
        'h1_resprate_max',
        'h1_resprate_min',
        'h1_spo2_max',
        'h1_spo2_min',
        'h1_sysbp_max',
        'h1_sysbp_min',
        'h1_sysbp_noninvasive_max',
        'h1_sysbp_noninvasive_min',
        'd1_glucose_max',
        'd1_glucose_min',
        'apache_4a_hospital_death_prob',
        'apache_4a_icu_death_prob',
        'aids',
        'hepatic_failure',
        'ethnicityIndex',
        'genderIndex',
        'icu_admit_sourceIndex',
        'icu_stay_typeIndex',
        'icu_typeIndex',
        'apache_3j_bodysystemIndex',
        'apache_2_bodysystemIndex']

vectorAssembler = VectorAssembler(
    inputCols = cols,
    outputCol = "features"
)
sparse_df = vectorAssembler.transform(df)

# Next we convert the target variable aka label into the required index format:
label_indexer = StringIndexer(
    inputCol = 'Death_Outcome',
    outputCol = "label"
)
label_fit = label_indexer.fit(sparse_df)
sparse_df = label_fit.transform(sparse_df)

# Next we split the data between train and test, we use the stratified sampling method to obtain samples that best represent the population.
train = sparse_df.sampleBy("Death_Outcome", fractions={'False': 0.6, 'True': 0.85}, seed=20000)
test = sparse_df.subtract(train)

# Now we construct the Logistic Regression model.
lr = LogisticRegression(
    maxIter=150, 
    regParam=0.0, 
    elasticNetParam=0.0, 
    featuresCol = "features",
    labelCol = "label"
)

fit = lr.fit(train)

# Utilising a GeneralizedLinearRegression model to plot coefficients against their respective feature to evaluate their importance and if needed alter the model.
# Setting the model:
model = GeneralizedLinearRegression(family="binomial", link="logit", featuresCol="features", labelCol="label", maxIter = 50, regParam = 0.0)

# Train model:
models = model.fit(train)

# Get summary of the model:

summary = models.summary
print(summary)

# Plotting Coefficient Distribution
coef = np.sort(fit.coefficients)
plt.plot(coef)
plt.title('Coefficients Distribution')
plt.xlabel('#')
plt.ylabel('Beta Coefficients')
plt.show()

# Making 5 predictions based on the train set.
train_results = fit.evaluate(train).predictions
train_results.show(5)

# Gathering the predictions and labels into a single dataframe from the trained model.
prediction_and_label_train = fit.transform(train).select("label", "prediction","probability")

# Calculating accuracy of the train model.
evaluator_t = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction"
)

acc_t = evaluator_t.evaluate(
    prediction_and_label_train, 
    {evaluator_t.metricName: "accuracy"}
)
print("Accuracy of the train model is: {:.3f}".format(acc_t))

# Calculating F1 score for the train model:
f1_t = evaluator_t.evaluate(
    prediction_and_label_train, 
    {evaluator_t.metricName: "f1"}
)
print("F1 Score of the train model is: {:.3f}".format(f1_t))

# Calculating Log loss for the train model:
logloss_t = evaluator_t.evaluate(
    prediction_and_label_train, 
    {evaluator_t.metricName: "logLoss"}
)
print("Log Loss of the train model is: {:.3f}".format(logloss_t))

# Creating a confusion matrix to collect our TP, TN, FP, FN of the train model:
predlabel_train = prediction_and_label_train.toPandas()
confusion_matrix_train = pd.crosstab(
    predlabel_train["label"], 
    predlabel_train["prediction"], 
    rownames=["Actual"], 
    colnames=["Predicted"]
)
confusion_matrix_train

# Calculating Sensitivity and Specificity of the train model:
TN = confusion_matrix_train[0][0]
FN = confusion_matrix_train[0][1]
TP = confusion_matrix_train[1][1]
FP = confusion_matrix_train[1][0]
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print("Train Sensitivity: {:.3f}\nTrain Specificity: {:.3f}".format(sensitivity,specificity))

# Gathering the predictions and labels into a single dataframe for the test set.
prediction_and_label = fit.transform(test).select("label", "prediction","probability")

# Making 5 predictions based on the test set.
prediction_and_label.show(5)

# Next we retrieve and print resuls for Accuracy, F1 score, Precision, Recall and Log-Loss for the test model.

# Calculating accuracy.
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction"
)

acc = evaluator.evaluate(
    prediction_and_label, 
    {evaluator.metricName: "accuracy"}
)
print("Accuracy of the test model is: {:.3f}".format(acc))

# Calculating weighted precision:
precision = evaluator.evaluate(
    prediction_and_label, 
    {evaluator.metricName: "weightedPrecision"}
)
print("Weighted Precision of the test model is: {:.3f}".format(precision))

# Calculating weighted recall:
recall = evaluator.evaluate(
    prediction_and_label, 
    {evaluator.metricName: "weightedRecall"}
)
print("Weighted Recall of the test model is: {:.3f}".format(recall))

# Calculating F1 score:
f1 = evaluator.evaluate(
    prediction_and_label, 
    {evaluator.metricName: "f1"}
)
print("F1 Score of the test model is: {:.3f}".format(f1))

# Calculating Log loss:
logloss = evaluator.evaluate(
    prediction_and_label, 
    {evaluator.metricName: "logLoss"}
)
print("Log Loss of the test model is: {:.3f}".format(logloss))

# Creating a confusion matrix to collect our TP, TN, FP, FN:
predlabel = prediction_and_label.toPandas()
confusion_matrix = pd.crosstab(
    predlabel["label"], 
    predlabel["prediction"], 
    rownames=["Actual"], 
    colnames=["Predicted"]
)
confusion_matrix

# Visualising the confusion matrix as a Seaborn heatmap.
fig = sns.heatmap(confusion_matrix, annot=True, fmt="d", cbar=False)
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
plt.title('Unregularised Model: Actauls vs Predicted')
plt.savefig("confusion_matrix.png")

# Retrieving Mathew's Correlation Coefficient:
mcc = matthews_corrcoef(predlabel["label"],predlabel["prediction"])
print("Matthews Correlation Coefficient: {:.3f}".format(mcc))

# Retrieving Cohen's Kappa:
kappa = cohen_kappa_score(predlabel["label"],predlabel["prediction"])
print("Cohen's Kappa: {:.3f}".format(kappa))

# Calculating Sensitivity and Specificity:
TN = confusion_matrix[0][0]
FN = confusion_matrix[0][1]
TP = confusion_matrix[1][1]
FP = confusion_matrix[1][0]
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print("Sensitivity of the test model is: {:.3f}\nSpecificity of the test model is: {:.3f}".format(sensitivity,specificity))

# This is not the accurate representation of the AUC score, keeping code for educational purposes.
from numpy.lib.function_base import average
# Calculating 'Area Under the ROC' curve
results = fit.transform(test)
my_eval = BinaryClassificationEvaluator(labelCol='label',
                                        rawPredictionCol='prediction',
                                        metricName="areaUnderROC")
results.select('label','prediction')
AUC = my_eval.evaluate(results)
print("AUC score is : ",AUC)

# Fiting the logistic regression model to the test set to plot and calculate AreaUnderROC & Precision-Recall Curve
final_lr = lr.fit(test)

# Plotting and calculating AreaUnderROC
plt.figure(figsize=(5,5))
plt.plot([0, 1], [0, 1], 'r--')
plt.plot(final_lr.summary.roc.select('FPR').collect(),
         final_lr.summary.roc.select('TPR').collect())
plt.title('Receiver Operating Characteristic Curve - Test Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

print('\n','Test set areaUnderROC: ' + str(round(final_lr.summary.areaUnderROC, 4)))

# Plotting and calculating Precision-Recall Curve
plt.figure(figsize=(5,5))
plt.plot([0, 1], [0, 1], 'r--')
plt.plot(final_lr.summary.pr.select('precision').collect(),
         final_lr.summary.pr.select('recall').collect())
plt.title('Precision-Recall Curve - Test Model')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

"""Regularised Logistic Regression Model:"""

print('Overall the results are very promising in the first run of the Logistic Regression, however we always look to imporve \n',
      'We will look to reguralise the model in an aim to imporve overall scores.')

# We will start with elastic net reguralization which is the L1 and L2 regularization balance.
lr_R = LogisticRegression(
    maxIter=150, 
    regParam=0.5, 
    elasticNetParam=0.1,
    featuresCol = "features",
    labelCol = "label"
    )

# Fit the model
lrModel = lr_R.fit(train)

# Gathering the predictions and labels into a single dataframe from the trained regularised model.
prediction_and_label_Rtrain = fit.transform(train).select("label", "prediction","probability")

# Calculating accuracy of the regularised train model.
evaluatorT = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction"
)

accT = evaluatorT.evaluate(
    prediction_and_label_Rtrain, 
    {evaluatorT.metricName: "accuracy"}
)
print("Accuracy of the regularised train model is: {:.3f}".format(accT))

# Calculating F1 score for the train model:
f1T = evaluatorT.evaluate(
    prediction_and_label_Rtrain, 
    {evaluatorT.metricName: "f1"}
)
print("F1 Score of the regularised train model is: {:.3f}".format(f1T))

# Calculating Log loss for the train model:
loglossT = evaluatorT.evaluate(
    prediction_and_label_Rtrain, 
    {evaluatorT.metricName: "logLoss"}
)
print("Log Loss of the regularised train model is: {:.3f}".format(loglossT))

# Creating a confusion matrix to collect our TP, TN, FP, FN of the regularised train model:
predlabel_Rtrain = prediction_and_label_Rtrain.toPandas()
confusion_matrix_Rtrain = pd.crosstab(
    predlabel_Rtrain["label"], 
    predlabel_Rtrain["prediction"], 
    rownames=["Actual"], 
    colnames=["Predicted"]
)
confusion_matrix_Rtrain

# Calculating Sensitivity and Specificity of the train model:
TN_T = confusion_matrix_Rtrain[0][0]
FN_T = confusion_matrix_Rtrain[0][1]
TP_T = confusion_matrix_Rtrain[1][1]
FP_T = confusion_matrix_Rtrain[1][0]
sensitivity = TP_T / (TP_T + FN_T)
specificity = TN_T / (TN_T + FP_T)
print("Regularised Train Model Sensitivity: {:.3f}\nRegularised Train Model Specificity: {:.3f}".format(sensitivity,specificity))

# Gathering the predictions and labels into a single dataframe for the regularised test set.
prediction_and_label2 = lrModel.transform(test).select("label", "prediction","probability")

# Calculating Accuracy:
evaluator2 = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction"
)

acc2 = evaluator2.evaluate(
    prediction_and_label2, 
    {evaluator2.metricName: "accuracy"}
)
print("Accuracy of the regularised test model is: {:.3f}".format(acc2))

# Calculating Weighted Precision
precision2 = evaluator2.evaluate(
    prediction_and_label2, 
    {evaluator2.metricName: "weightedPrecision"}
)
print("Weighted Precision of the regularised test model is: {:.3f}".format(precision2))

# Calculating Weighted Recall
recall2 = evaluator2.evaluate(
    prediction_and_label2, 
    {evaluator2.metricName: "weightedRecall"}
)
print("Weighted Recall of the regularised test model is: {:.3f}".format(recall2))

# Calculating F1
f1_2 = evaluator2.evaluate(
    prediction_and_label2, 
    {evaluator2.metricName: "f1"}
)
print("F1 Score of the regularised test model is: {:.3f}".format(f1_2))

# Calculating log Loss
logloss2 = evaluator2.evaluate(
    prediction_and_label2, 
    {evaluator2.metricName: "logLoss"}
)
print("Log Loss of the regularised test model is: {:.3f}".format(logloss2))

# Creating a confusion matrix to collect our TP, TN, FP, FN from our test regularised model:
predlabel2 = prediction_and_label2.toPandas()
confusion_matrix2 = pd.crosstab(
    predlabel2["label"], 
    predlabel2["prediction"], 
    rownames=["Actual"], 
    colnames=["Predicted"]
)
confusion_matrix2

# Visualising the confusion matrix as a Seaborn heatmap.
fig = sns.heatmap(confusion_matrix2, annot=True, fmt="d", cbar=False)
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
plt.title('Regularised Model: Actauls vs Predicted')
plt.savefig("confusion_matrix.png")

# Calculating Sensitivity and Specificity for our test regularised model:
TN2 = confusion_matrix2[0][0]
FN2 = confusion_matrix2[0][1]
TP2 = confusion_matrix2[1][1]
FP2 = confusion_matrix2[1][0]
sensitivity2 = TP2 / (TP2 + FN2)
specificity2 = TN2 / (TN2 + FP2)
print("Regularised Test Model Sensitivity: {:.3f}\nRegularised Test Model Specificity: {:.3f}".format(sensitivity2,specificity2))

# Plotting and calculation AreaUnderROC for the regularised model
final_lr_2 = lr_R.fit(test)

plt.figure(figsize=(5,5))
plt.plot([0, 1], [0, 1], 'r--')
plt.plot(final_lr_2.summary.roc.select('FPR').collect(),
         final_lr_2.summary.roc.select('TPR').collect())
plt.title('Receiver Operating Characteristic Curve - Regularised Test Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

print('\n','Regularised Test set areaUnderROC: ' + str(round(final_lr_2.summary.areaUnderROC, 2)), '\n')

print("The full code was executed in: ","--- %s seconds ---" % (time.time() - start_time))


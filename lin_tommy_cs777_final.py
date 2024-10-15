# Import all dependencies/necessary libraries
import numpy as np

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, NaiveBayes


# Finds accuracy given a dataframe with predictions and labels
def accuracy(preds):
    num_correct = preds.filter("label==prediction").count()
    total = preds.count()
    return num_correct / total

# Finds precision (avg one vs rest) given predictions and labels
def precision(cm, classes):
    precisions = []
    # Calculates precision using the columns of the consfusion matrix
    for i in classes:
        preds_i = cm[:,i]
        num_preds_i = np.sum(preds_i)
        tp_i = preds_i[i]
        fp_i = num_preds_i - tp_i
        precisions.append(tp_i / (tp_i + fp_i))
    return precisions, np.mean(precisions)

# Finds recall (avg one vs rest) given predictions and labels
def recall(cm, classes):
    recalls = []
    # Calculates the recall using the rows of the confusion matrix
    for i in classes:
        preds_i = cm[i]
        num_preds_i = np.sum(preds_i)
        tp_i = preds_i[i]
        fn_i = num_preds_i - tp_i
        recalls.append(tp_i / (tp_i + fn_i))
    return recalls, np.mean(recalls)

# Calculates F1 score given precision and recall
def f1(precision, recall):
    return (2 * precision * recall) / (precision + recall)

# Calculates the confusion matrix using the DataFrame of predictions and labels
# Index 0, 1 denotes a label of 0 and a prediction of 1
def confusion_matrix(preds, classes):
    cm = np.zeros((len(classes), len(classes)))
    for i in classes:
        preds_i = preds.filter(preds.label == i)
        for j in classes:
            cm[i][j] = preds_i.filter(preds.prediction == j).count()
    return cm

# Prints confusion matrix
def print_confusion_matrix(cm, classes):
    # Prints column headers of confusion matrix (predictions)
    print("        |", end="")
    for i in classes:
        print(f"{f'Pred {i}':^8}|", end="")
    print("")

    # Prints row headers of confusion matrix (labels)
    for i in classes:
        print(f"Label {i} |", end="")
        # Prints values in each row
        for num in cm[i].astype(int).astype(str):
            print(f"{ num:^8}|", end="")
        print("")


if __name__ == "__main__":


    ### LOAD THE DATA

    # Create Spark instances to work with PySpark
    sc = SparkContext(appName="CS777 Final Project: Multi-Classification")
    ss = SparkSession.builder.getOrCreate()

    # Load in the data from the csv file into a PySpark DataFrame
    df = ss.read.format("csv").options(header="true").load("text.csv").cache()
    df = df.toDF("id", "text", "label")
    df = df.withColumn("label", df["label"].cast("int"))

    # Create dictionary of class names to refer to
    class_names = {
        0: "sad",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }


    ### PREPROCESS THE DATA

    # Convert the text into lowercase and remove numbers and remove 1 to 2 character words
    df = df.withColumn("preprocessed_text", F.regexp_replace("text", r"[^a-zA-Z ]", " "))
    df = df.withColumn("preprocessed_text", F.lower(F.col("text")))
    df = df.withColumn("preprocessed_text", F.regexp_replace("preprocessed_text", r"\b\w{1,2}\b", ""))

    # Create Pipeline for transforming textual data
    tokenizer = Tokenizer(inputCol="preprocessed_text", outputCol="tokens")
    stop_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="sw_removed")
    blank_remover = StopWordsRemover(inputCol=stop_remover.getOutputCol(), outputCol="blank_removed", stopWords=[""])
    counter = CountVectorizer(inputCol=blank_remover.getOutputCol(), outputCol="counts", vocabSize=1000)
    idf = IDF(inputCol=counter.getOutputCol(), outputCol="tfidf")
    pre_pipeline = Pipeline(stages=[tokenizer, stop_remover, blank_remover, counter, idf])

    # Split DataFrame into training and testing set with a 80/20 split
    train, test = df.randomSplit(weights=[0.8, 0.2], seed=777)
    print(f"Training Set: {train.count()}")
    train.groupBy("label").count().show()
    print(f"\nTesting Set: {test.count()}")
    test.groupBy("label").count().show()

    # Get list of classes
    classes = [i['label'] for i in train.select('label').distinct().collect()]
    classes = np.sort(classes)

    # Get total number of training rows
    train_size = train.count()

    # Assign weights to each class because dataset is imbalanced
    # Creates an empty weight column and iteratively adds the weight of each class
    train = train.withColumn("weight", F.lit(0))
    for i in classes:
        num_in_class = train.filter(f"label=={i}").count()
        weight = train_size / (len(classes) * num_in_class)
        train = train.withColumn("weight", F.when(F.col("label")==i, weight).otherwise(F.col("weight")))
        print(f"Class: {i} | # Class: {num_in_class:<6} | Weight: {weight:.4f}\n\n")

    # Fit the transform pipeline using the training set
    pre_model = pre_pipeline.fit(train)

    # Transform the data
    transformed_train = pre_model.transform(train).cache()
    transformed_test = pre_model.transform(test).cache()

    ### TRAIN THE MODELS

    # Create Random Forest Classifier model with 30 trees and depth of 30
    rf = RandomForestClassifier(numTrees=30, 
                                maxDepth=30, 
                                featuresCol="counts", 
                                labelCol="label", 
                                weightCol="weight", 
                                seed=777)
    
    # Train Random Forest Classifier model using the transformed training dataset
    rf_model = rf.fit(transformed_train)

    # Create Logistic Regression model 
    lr = LogisticRegression(maxIter=20, regParam=0.1, featuresCol="tfidf", weightCol="weight", labelCol="label")

    # Train Logistic Regression model using the transformed training dataset
    lr_model = lr.fit(transformed_train)

    # Create Support Vector Classifier model
    nb = NaiveBayes(modelType="multinomial", weightCol="weight", labelCol="label", featuresCol="tfidf")

    # Train Support Vector Classifier model using the transformed training dataset
    nb_model = nb.fit(transformed_train)


    ### EVALUATE THE MODELS
    rf_preds = rf_model.transform(transformed_test)
    rf_cm = confusion_matrix(rf_preds, classes)
    rf_precisions, rf_avg_precision = precision(rf_cm, classes)
    rf_recalls, rf_avg_recall = recall(rf_cm, classes)
    rf_accuracy = accuracy(rf_preds)
    print("Random Forest Classifier:")
    print_confusion_matrix(rf_cm, classes)
    print(f"Accuracy: {rf_accuracy:.4f}")
    print(f"Precision: {rf_avg_precision:.4f}, Recall: {rf_avg_recall:.4f}, F1: {f1(rf_avg_precision, rf_avg_recall):.4f}\n\n")


    lr_preds = lr_model.transform(transformed_test)
    lr_cm = confusion_matrix(lr_preds, classes)
    lr_precisions, lr_avg_precision = precision(lr_cm, classes)
    lr_recalls, lr_avg_recall = recall(lr_cm, classes)
    lr_accuracy = accuracy(lr_preds)
    print("Logistic Regression Classifier:")
    print_confusion_matrix(lr_cm, classes)
    print(f"Accuracy: {lr_accuracy:.4f}")
    print(f"Precision: {lr_avg_precision:.4f}, Recall: {lr_avg_recall:.4f}, F1: {f1(lr_avg_precision, lr_avg_recall):.4f}\n\n")


    nb_preds = nb_model.transform(transformed_test)
    nb_cm = confusion_matrix(nb_preds, classes)
    nb_precisions, nb_avg_precision = precision(nb_cm, classes)
    nb_recalls, nb_avg_recall = recall(nb_cm, classes)
    nb_accuracy = accuracy(nb_preds)
    print("Naive Bayes Classifier:")
    print_confusion_matrix(nb_cm, classes)
    print(f"Accuracy: {nb_accuracy:.4f}")
    print(f"Precision: {nb_avg_precision:.4f}, Recall: {nb_avg_recall:.4f}, F1: {f1(nb_avg_precision, nb_avg_recall):.4f}\n")


    ### GATHER ADDITIONAL INSIGHTS
    # Extract coefficients from the trained Logistic Regression Model
    coefs = lr_model.coefficientMatrix.toArray()

    # Extract vocabulary from the fitted Count Vectorizer
    vocab = pre_model.stages[3].vocabulary

    # Print out the most common words in the vocabulary
    print("Top 20 Most Common Words")
    print(vocab[:20])

    # Print out the most correlated and least correlated words for each class
    print("\nEmotion & Word Correlations")
    for i in range(len(coefs)):
        sorted_coef = np.argsort(coefs[i])
        top = sorted_coef[-10:]
        bottom = sorted_coef[:10]
        top_words = [vocab[i] for i in top]
        bottom_words = [vocab[i] for i in bottom]
        print(f"Top Words for {class_names[i]}: {top_words}")
        print(f"Bottom Words for {class_names[i]}: {bottom_words}\n")


    # Figure out which emotions were predicted the most after the correct emotion
    # Example: If model doesn't guess sad for a sad tweet, what did it confuse it with?
    print("Which Emotions were mistaken for another Emotion the most?")
    for i in range(len(lr_cm)):
        row = lr_cm[i]
        order = np.argsort(row)
        second = order[-2]
        print(f"Label '{class_names[i]}' most confused with '{class_names[second]}'")
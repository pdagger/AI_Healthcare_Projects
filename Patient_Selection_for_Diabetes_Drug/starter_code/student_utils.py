import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    # dict of codes to generic drug name
    code_drug = dict(zip(ndc_df.NDC_Code, ndc_df['Non-proprietary Name']))
    # the nan values was given a random drug equivalent
    code_drug[np.nan] = np.nan
    # new column for drug names and map codes to drug names
    df['generic_drug_name'] = df.ndc_code.map(code_drug)

    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    # sort df in ascending order by encounter_id
    df.sort_values('encounter_id', inplace=True)
    # chose first encounter number per patient
    first_encounter_values = df.groupby('patient_nbr')['encounter_id'].head(1).values
    # make df with first encounter data
    first_encounter_df = df[df.encounter_id.isin(first_encounter_values)].reset_index(drop = True)

    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    # set ratios
    validation_ratio = 0.2
    test_ratio = 0.2
    # randomize dataset
    df = df.iloc[np.random.permutation(len(df))]
    # define split sizes with respect to patient key
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    validation_size = round(total_values * validation_ratio)
    test_size = round(total_values * test_ratio)
    train_size = total_values - validation_size - test_size
    # split df
    train = df[df[patient_key].isin(unique_values[: train_size])].reset_index(drop=True)
    validation = df[df[patient_key].isin(unique_values[train_size: train_size + validation_size])].reset_index(drop=True)
    test = df[df[patient_key].isin(unique_values[train_size + validation_size:])].reset_index(drop=True)

    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        # read vocab from file
        vocab = tf.feature_column.categorical_column_with_vocabulary_file(\
                                            key=c, vocabulary_file=vocab_file_path, num_oov_buckets=1)
        # create one-hot encoding from vocab
        tf_categorical_feature_column = tf.feature_column.indicator_column(vocab)

        output_tf_list.append(tf_categorical_feature_column)

    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key=col, \
                                                          default_value=default_value, \
                                                          normalizer_fn=normalizer, \
                                                          dtype=tf.float64)

    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()

    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    min_days = 5 # minimum days to stay at hospital
    student_binary_prediction = df[col].apply(lambda x: 1 if x > min_days else 0)

    return student_binary_prediction

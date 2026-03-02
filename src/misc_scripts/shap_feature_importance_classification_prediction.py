## Import SHAP library
import shap

## Load JS visualization code to notebook
shap.initjs() # you need this so the plots can be displayed

## Also go ahead and import pandas for the functions we'll write
import pandas as pd

## Split the data into training and test sets prior to preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=random_seed)

## Make a list of all columns that are currently object dtype
cat_cols = list(X_train.select_dtypes('O').columns)

## Create a pipeline for one hot encoding categorical columns
cat_transformer = Pipeline(steps = [
  ('ohe', OneHotEncoder(handle_unknown='error', 
                        sparse=False,
                        drop='if_binary'))])

## Define pipeline for preprocessing X
preprocessing = ColumnTransformer(transformers=[
    ('cat', cat_transformer, cat_cols)
])

## Preprocess training and test predictors (X)
X_train_tf = preprocessing.fit_transform(X_train)
X_test_tf = preprocessing.transform(X_test)

## Get the feature names in the order they appear in preprocessed data
feature_names = preprocessing.named_transformers_['cat'].named_steps['ohe'].get_feature_names(cat_cols)

## Store the best fitted classifier and its booster
binary_est = xgb_bin_grid.best_estimator_ # from tuning with GridSearchCV
binary_model = binary_est.named_steps['xgb'].get_booster()

## Convert transformed (preprocessed) X train set into pandas DataFrame
X_train_df = pd.DataFrame(X_train_tf, columns=feature_names)

## Convert transformed (preprocessed) X train set into pandas DataFrame
X_train_df = pd.DataFrame(X_train_tf, columns=feature_names)

## Calculate SHAP values for model
binary_explainer = shap.TreeExplainer(binary_model)
binary_shap_values = binary_explainer.shap_values(X_train_df)


## Print model prediction, true label, and shap force plot for third row
  ## of training set
shap_force(binary_est, 
           'xgb', # name of fitted classifier step in pipeline
           2, X_train_df, y_train, 
           binary_explainer, 
           binary_shap_values)


def shap_force(clf, clf_step_name, index, 
               X_train_df, y_train,
               explainer, shap_vals):
  
    """Takes in a fitted classifier Pipeline, the name of the classifier step,
        the X training DataFrame, the y train array, a shap explainer, and the
        shap values to print the ground truth and predicted label and display
        the shap force plot for the record specified by index.
    Args:
        clf (estimator): An sklearn Pipeline with a fitted classifier as the final step.
        clf_step_name (str): The name given to the classifier step of the pipe.
        X_train_df (DataFrame): A Pandas DataFrame from the train-test-split
            used to train the classifier, with column names corresponding to
            the feature names.
        y_train (series or array): Subset of y data used for training.
        index (int): The index of the observation of interest.
        explainer (shap explainer): A fitted shap.TreeExplainer object.
        shap_vals (array): The array of shap values.
    Returns:
        Figure: Shap force plot showing the breakdown of how the model made
            its prediction for the specified record in the training set.
    """    
    
    
    ## Store model prediction and ground truth label
    pred = clf.named_steps[clf_step_name].predict(X_train_df.iloc[index,:])
    true_label = y_train.iloc[index]
    
    
    ## Assess accuracy of prediction
    if true_label == pred:
        accurate = 'Correct!'
    else:
        accurate = 'Incorrect'
    
    
    ## Print output that checks model's prediction against true label
    print('***'*12)
    # Print ground truth label for row at index
    print(f'Ground Truth Label: {true_label}')
    print()
    # Print model prediction for row at index
    print(f'Model Prediction:  {pred} -- {accurate}')
    print('***'*12)
    print()
    
    
    ## Plot the prediction's explanation
    fig = shap.force_plot(explainer.expected_value,
                              shap_vals[index,:],
                              X_train_df.iloc[index,:])
    
    
    return fig

#force plots for multi-class classification
from sklearn.preprocessing import LabelEncoder

## Preprocess training and test target (y) after having performed train-test split
le = LabelEncoder()
y_multi_train = pd.Series(le.fit_transform(y_multi_train))
y_multi_test = pd.Series(le.transform(y_multi_test))

## Check classes
le.classes_

## Store the best fitted classifier and its booster
multi_est = xgb_multi_grid.best_estimator_ # from tuning with GridSearchCV
multi_model = multi_est.named_steps['xgb'].get_booster()

## Convert transformed (preprocessed) X train set into pandas DataFrame
X_train_df = pd.DataFrame(X_train_tf, columns=feature_names)

## Calculate SHAP values for model
multi_explainer = shap.TreeExplainer(multi_model)
multi_shap_values = multi_explainer.shap_values(X_train_df)

## Check the type of the output when calculating SHAP values for a multi-class target
print(type(multi_shap_values)) # output: <class 'list'>

## Check the type of the first item in the list
print(type(multi_shap_values[0])) # output: <class 'numpy.ndarray'>

## Check the number of items in the list
print(len(multi_shap_values)) # output: 3

## Print model prediction, true label, and shap force plots
  ## for third row of training set
multi_shap_force(multi_est, 'xgb', 2,
                 X_train_df, y_train,
                 multi_explainer,
                 multi_shap_values,
                 classes='all')

def multi_shap_force(clf, clf_step_name, index,
                     X_train_df, y_train,
                     explainer, multi_shap_vals,
                     classes='all'):
  
    """Takes in a fitted classifier Pipeline, the name of the classifier step,
        the X training DataFrame, the y train array, a shap explainer, and the
        multiclass shap values. Prints the ground truth and predicted label for
        the record of interest and displays shap force plots of the desired classes
        for the record specified by index.
    Args:
        clf (estimator): An sklearn Pipeline with a fitted classifier as the final step.
        clf_step_name (str): The name given to the classifier step of the pipe.
        index (int): The index of the observation of interest.
        X_train_df (DataFrame): A Pandas DataFrame that from the train-test-split
            used to train the classifier, with column names corresponding to
            the feature names.
        y_train (series or array): Subset of y data used for training.
        explainer (shap explainer): A fitted shap.TreeExplainer object.
        multi_shap_vals (list): The list of arrays of shap values. One array per 
            target label.
        classes (str, optional): A string specifying which shap force plots
            to display for the specified record. Options are 'all' (displays for all
            class labels), 'true' (displays only the plot for the ground truth label for
            the record), 'pred' (displays only the plot for the predicted label for
            the record), or 'both' (displays both 'true' and 'pred'). Defaults to 'all'.
    """

    ## Create dict for mapping class labels
    label_dict = {0: 'Early',
                  1: 'Election Day',
                  2: 'No Vote'}

    
    ## Store model prediction and ground truth label for specified index
    pred = int(clf.named_steps[clf_step_name].predict(X_train_df.iloc[index,:]))
    true_label = pd.Series(y_train).iloc[index]


    ## Assess accuracy of prediction
    if true_label == pred:
        accurate = 'Correct!'
    else:
        accurate = 'Incorrect'
        

    ## Print output that checks model's prediction against true label
    print('***'*17)
    # Print ground truth label for row at index
    print(f'Ground Truth Label: {true_label} - {label_dict[true_label]}')
    print()
    # Print model prediction for row at index
    print(f'Model Prediction:  [{pred}] - {label_dict[pred]} -- {accurate}')
    print('***'*17)
    print()
    print()
 
    
    ## Determine which classes to show force plots for
    # All classes 
    if classes == 'all':
        ## Visualize the ith prediction's explanation for all classes
        print('Early Vote Class (0)')
        display(shap.force_plot(explainer.expected_value[0],
                    multi_shap_vals[0][index],
                    X_train_df.iloc[index,:]))
        print()

        print('Election Day Vote Class (1)')
        display(shap.force_plot(explainer.expected_value[1],
                    multi_shap_vals[1][index],
                    X_train_df.iloc[index,:]))
        print()

        print('No Vote Class (2)')
        display(shap.force_plot(explainer.expected_value[2],
                    multi_shap_vals[2][index],
                    X_train_df.iloc[index,:]))
        
    
    # Only the class predicted by the model
    elif classes == 'pred':
        print(f'Predicted: {label_dict[pred]} Class {pred}')
        display(shap.force_plot(explainer.expected_value[pred],
                                multi_shap_vals[pred][index],
                                X_train_df.iloc[index,:]))
    
    
    # Only the ground truth label
    elif classes == 'true':
        print(f'True: {label_dict[true_label]} Class {true_label}')
        display(shap.force_plot(explainer.expected_value[true_label],
                    multi_shap_vals[true_label][index],
                    X_train_df.iloc[index,:]))
    
    
    # Both the predicted and ground truth (identical plots if prediction is correct)
    elif classes == 'both':
        print(f'Predicted: {label_dict[pred]} Class {pred}')
        display(shap.force_plot(explainer.expected_value[pred],
                                multi_shap_vals[pred][index],
                                X_train_df.iloc[index,:]))
        print()

        print(f'True: {label_dict[true_label]} Class {true_label}')
        display(shap.force_plot(explainer.expected_value[true_label],
                    multi_shap_vals[true_label][index],
                    X_train_df.iloc[index,:]))
view rawmulti_shap_force_func_hc.py hosted with ❤ by Git

def multi_shap_force_le(clf, clf_step_name, index,
                        X_train_df, y_train,
                        explainer, multi_shap_vals,
                        le_classes,
                        classes='both'):
  
    """Takes in a fitted classifier Pipeline, the name of the classifier step,
        the X training DataFrame, the y train array, a shap explainer, and the
        multiclass shap values to print the ground truth and predicted label for
        the record and display shap force plots of the desired classes
        for the record specified by index.
    Args:
        clf (estimator): An sklearn Pipeline with a fitted classifier as the final step.
        clf_step_name (str): The name given to the classifier step of the pipe.
        index (int): The index of the observation of interest.
        X_train_df (DataFrame): A Pandas DataFrame that from the train-test-split
            used to train the classifier, with column names corresponding to
            the feature names.
        y_train (series or array): Subset of y data used for training.
        explainer (shap explainer): A fitted shap.TreeExplainer object
        multi_shap_vals (list): The list of arrays of shap values. One array per 
            target label.
        le_classes (array): The classes_ attribute of the label encoded target variable.
        classes (str, optional): A string specifying which shap force plots
            to display for the specified record. Options are 'all' (displays for all
            class labels), 'true' (displays only the plot for the ground truth label for
            the record), 'pred' (displays only the plot for the predicted label for
            the record), or 'both' (displays both 'true' and 'pred'). Defaults to 'both'.
    """

    ## Create dict for mapping class labels
    label_dict = {}
    for i, label in list(enumerate(le_classes)):
        label_dict[i] = label
        
    ## Store model prediction and ground truth label for that index
    pred = int(clf.named_steps[clf_step_name].predict(X_train_df.iloc[index,:]))
    true_label = pd.Series(y_train).iloc[index]


    ## Assess accuracy of prediction
    if true_label == pred:
        accurate = 'Correct!'
    else:
        accurate = 'Incorrect'
        

    ## Print output that checks model's prediction against true label
    print('***'*17)
    # Print ground truth label for row at index
    print(f'Ground Truth Label: {true_label} - {label_dict[true_label]}')
    print()
    # Print model prediction for row at index
    print(f'Model Prediction:  [{pred}] - {label_dict[pred]} -- {accurate}')
    print('***'*17)
    print()
    print()
    
 
    ## Determine which classes to show force plots for
    # All classes 
    if classes == 'all':
        ## Visualize the ith prediction's explanation for all classes
        for key in range(len(label_dict)):
            print(f'{label_dict[key]} Class ({key})')
            display(shap.force_plot(explainer.expected_value[key],
                        multi_shap_vals[key][index],
                        X_train_df.iloc[index,:]))
            print()
    
    
    # Only the class predicted by the model
    elif classes == 'pred':
        print(f'Predicted: {label_dict[pred]} Class {pred}')
        display(shap.force_plot(explainer.expected_value[pred],
                                multi_shap_vals[pred][index],
                                X_train_df.iloc[index,:]))

     
    # Only the ground truth label
    elif classes == 'true':
        print(f'True: {label_dict[true_label]} Class {true_label}')
        display(shap.force_plot(explainer.expected_value[true_label],
                    multi_shap_vals[true_label][index],
                    X_train_df.iloc[index,:]))

    
    # Both the predicted and ground truth (identical plots if prediction is correct)
    elif classes == 'both':
        print(f'Predicted: {label_dict[pred]} Class {pred}')
        display(shap.force_plot(explainer.expected_value[pred],
                                multi_shap_vals[pred][index],
                                X_train_df.iloc[index,:]))
        print()

        print(f'True: {label_dict[true_label]} Class {true_label}')
        display(shap.force_plot(explainer.expected_value[true_label],
                    multi_shap_vals[true_label][index],
                    X_train_df.iloc[index,:]))
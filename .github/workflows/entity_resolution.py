import pandas as pd
import sklearn.linear_model
import sklearn.ensemble
import sklearn.model_selection
import sklearn.preprocessing
import pickle
import os
import json

class ReusableClassifier(): 
    def __init__(self, model_type: str):
        """create a classifier, storing a model and metadata

        Args:
            model_type (str): can include random forests, logistic regression
        """
        
        
        if model_type == "logistic_regression":
            self.model = self._create_logistic_regression
        elif model_type == "random_forest":
            self.model = self._create_random_forest
        else:
            raise ValueError
        
        self.metadata['model_type'] = self.model
        #initializing the scaler var for use in the future
        #add all vars to the init function
        #and all SHARED vars
        self.scaler = None
        
    def train(self, features: pd.DataFrame, labels: pd.Series, test_frac: float = 0.1):
        
        self._assess_tf_fraction(labels)
        
        #We need to scale the data
        #1. we can set the min to 0 and max to 1
        #we need to consider outliers
        #remmove outliers entirely
        # Use standardization by standard normal 
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(features)
        features = self.scaler.fit(features)
        
        """train the model from pandas data 

        Args:
            features (pd.DataFrame): input features and a dataframe
            labels (pd.Series): input labels
            test_frac (float, optional): Fraction of data to preserve for testing. Defaults to 0.1.
            X is another term for features, y is another term for labels
        """
        features_train, features_test, labels_train, labels_test = \
            sklearn.model_selection.train_test_split(features, labels, test_size = test_frac)
        self.model.fit(features_train, labels_train)
        pred_labels = self.model.predict(features_test)
        
        #manual accuracy
        accuracy = (pred_labels == labels_test).sum()/len(labels_test)
        accuracy = (pred_labels == labels_test).mean()
        
        self.metadata = {}
        self.metadate['training_rows'] = len(features_train)
        self.metadata['accuracy'] = accuracy
        print(f"accuracy test set is: {self.metadata['accuracy']}")
        
        
        
    def predict(self, features: pd.DataFrame):
        self.scaler.transform(features)
        self.model.predict(features)
    
    def save(self, path:str):
        #data/super_cool_model.super_stupid_ext
        model_path, _ = os.path.splitext(path)
        scaler_path, _ = model_path + '_scaler.pkl'
        metadata_path, _ = model_path + 'json'
        model_path = model_path + ".pkl"
        
        with open(model_path, 'wb') as fp:
            pickle.dump(self.model, fp)
        with open(scaler_path, 'wb') as fp:
            pickle.dump(self.scaler, fp)
        with open(metadata_path, 'w') as fp:
            json.dump(self.metadata, fp)
    
    def load(self, path:str):
        with open(model_path, 'rb') as fp:
            self.model = pickle.load(fp)
        with open(scaler_path, 'rb') as fp:
            self.scaler = pickle.load(fp)
        with open(metadata_path, 'r') as fp:
            self.metadata = json.load(fp)
      
        
    def _assess_tf_fraction(self, labels: pd.Series):
        """throw an error for dramatically un-weighted data

        Args:
            labels (pd.Series): features to look at
        """
        if labels.sum() > 0.8*len(labels):
            raise ValueError("Too many trues")
        elif labels.sum() < .2*len(labels):
            raise ValueError("Too many falses")
        



    def _create_logistic_regression(self):
        """create a new logistic regression model from sklearn.
        """
        return sklearn.linear_model.LogisticRegression()
    
    
    
    def _create_random_forest(self):
        """
        Create a new random forest model from SKlearn and train the model
        """
        return sklearn.ensemble.RandomForestClassifier()
    
    
if __name__ == "__main__":
    import duq-lab-25.winequality
    wq = duq-lab-25.winequality
    wq.read(data/wine_quality.zip)
    df = wq.df
    labels = df['quality'] > 5
    features = df[['fixed acidity', 'sulfates', 'alcohol']]
    
    lr = ReusableClassifier('logistic_regression')
    lr.train(features, labels)
    
    rf = ReusableClassifier['random_forest']
    rf.train(features, labels)
    
    rf.save('data/model_rf')
    rf.load('data/model_rf')
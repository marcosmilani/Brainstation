from metaflow import FlowSpec, step, Flow, current, Parameter, IncludeFile, card, current
from metaflow.cards import Table, Markdown, Artifact

# Done (v0) move your labeling function from earlier in the notebook here
# labeling_function = lambda row: 0
def labeling_function(row):
    """
    A function to derive labels from the user's review data.
    This could use many variables, or just one. 
    In supervised learning scenarios, this is a very important part of determining what the machine learns!
   
    A subset of variables in the e-commerce fashion review dataset to consider for labels you could use in ML tasks include:
        # rating: Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
        # recommended_ind: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.
        # positive_feedback_count: Positive Integer documenting the number of other customers who found this review positive.

    In this case, we are doing sentiment analysis. 
    To keep things simple, we use the rating only, and return a binary positive or negative sentiment score based on an arbitrarty cutoff. 
    """
    # Done (v0): Add your logic for the labelling function here
    # It is up to you on what value to choose as the cut off point for the postive class
    # A good value to start would be 4
    # This function should return either a 0 or 1 depending on the rating of a particular row
    
    # Try catching some invalid values
    # if row.rating.dtype not in ('int64', 'float64'):
    #     return 0
    
    if row.rating >= 4:
        return 1
    else:
        return 0


class BaselineNLPFlow(FlowSpec):

    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter('split-sz', default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile('data', default='../data/Womens Clothing E-Commerce Reviews.csv')

    @step
    def start(self):

        # Step-level dependencies are loaded within a Step, instead of loading them 
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io 
        from sklearn.model_selection import train_test_split

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels 
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df['review_text'] = df['review_text'].astype('str')
        _has_review_df = df[df['review_text'] != 'nan']
        reviews = _has_review_df['review_text']
        labels = _has_review_df.apply(labeling_function, axis=1)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        self.df = pd.DataFrame({'label': labels, **_has_review_df})
        del df
        del _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({'review': reviews, 'label': labels})
        
        "Split the data into training (fit), validation (trial run), test (final exam) sets"
        SEED = 89
        _test_ratio = 0.4 # Default to 40% of the non-training data to be test set (train:val:test == 0.8:0.12:0.08)
        
        def train_validation_test_split (
            X, y, train_ratio: float, validation_ratio: float, test_ratio: float
        ):
            # Split up dataset into train and test, of which we split up the test.
            X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(1- train_ratio), random_state=SEED
            )

            # Split up test into two (validation and test).
            X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=(test_ratio / (test_ratio + validation_ratio)), random_state=SEED,
            )

            # Return the splits
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        # features (X), label (y)
        X = _df.iloc[:, ~_df.columns.isin(['label'])]
        y = _df[['label']]

        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = train_validation_test_split(
         X, y, 1.0-self.split_size, (1.0-_test_ratio)*(1.0-self.split_size), _test_ratio*(1.0-self.split_size)
        )

        print(f'num of rows in train set: {self.X_train.shape[0]}')
        print(f'num of rows in validation set: {self.X_val.shape[0]}')
        print(f'num of rows in test set: {self.X_test.shape[0]}')

        self.next(self.baseline)

    @step
    def baseline(self):
        # from sklearn.tree import DecisionTreeRegressor
        # from sklearn.ensemble import RandomForestRegressor
        # from sklearn.pipeline import Pipeline
        # from sklearn.preprocessing import MinMaxScalar, OneHotEncoder
        # from sklearn.compose import make_column_transformer
        # from sklearn.svm import SVC
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
        import pandas as pd
        
        "Numerical features"
        _num_features_ls = self.X_train.select_dtypes(include=['int64', 'float64']).columns.values
        "Categorical features"
        _cat_features_ls = self.X_train.select_dtypes(include='category').columns.values

        "Compute the baseline"
        # Basic baseline can be as simple as predicting the most frequent class for a classification problem
        # _pct_positive_sentiment = self.traindf.labels.sum() / self.traindf.labels.shape[0]

        dummy_model = DummyClassifier(strategy="most_frequent")
        dummy_model.fit(self.X_train, self.y_train)
        
        self.model = dummy_model

        ### Done: Fit and score a baseline model on the data, log the acc and rocauc as artifacts.
        self.base_acc = dummy_model.score(self.X_train, self.y_train)
        self.base_rocauc = 0.5 # AUC will be 0.5 for a dummy or random baseline model
        # self.base_rocauc = roc_auc_score(y_train, dummy_y_predict)

        self.next(self.end)
        
    @card(type='corise') # done: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):
        from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
        import pandas as pd
        from metaflow.cards import Table, Markdown, Artifact
        
        msg = 'Baseline Accuracy: {}\nBaseline AUC: {}'
        print(msg.format(
            round(self.base_acc,3), round(self.base_rocauc,3)
        ))

        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.base_acc))

        # Some playing around with the model artifacts
        dummy_model = self.model
        dummy_y_predict = dummy_model.predict(self.X_val)
        # Predict probabilities for the validation set (X_test)
        dummy_y_pred_prob = dummy_model.predict_proba(self.X_val)[:, 1]
        # Calculate the false positive rate, true positive rate, and thresholds using roc_curve
        fpr, tpr, thresholds = roc_curve(self.y_val, dummy_y_pred_prob)
        
        current.card.append(Markdown("## Examples of False Positives"))
        # Done: compute the false positive predictions where the baseline is 1 and the valid label is 0. 
        # Create a df with 2 cols: actual y, pred y for the training set
        _fp_df = pd.DataFrame({'predicted': list(dummy_y_predict), 'actuals': self.y_val.label})
        _fp_df['is_fp'] = [1 if row.predicted == 1 and row.actuals == 0 else 0 for _, row in _fp_df.iterrows()]

        print(f'num false postives: {_fp_df.is_fp.sum()}')
        self.X_val_fp_df = pd.merge(self.X_val, _fp_df.query('is_fp == 1'), left_index=True, right_index=True, how='inner')

        # Done: display the false_positives dataframe using metaflow.cards
        # Documentation: https://docs.metaflow.org/api/cards#table
        if self.X_val_fp_df.shape[0] == 0:
            _fp = pd.DataFrame()
        else:
            _fp = self.X_val_fp_df.sample(5)

        current.card.append(Table.from_dataframe(_fp))
            

        current.card.append(Markdown("## Examples of False Negatives"))
        # Done: compute the false negatives predictions where the baseline is 0 and the valdf label is 1. 
        _fn_df = _fp_df.copy()
        _fn_df['is_fn'] = [1 if row.predicted == 0 and row.actuals == 1 else 0 for _, row in _fn_df.iterrows()]
        self.X_val_fn_df = pd.merge(self.X_val, _fn_df.query('is_fn == 1'), left_index=True, right_index=True, how='inner')

        if self.X_val_fn_df.shape[0] == 0:
            _fn = pd.DataFrame()
        else:
            _fn = self.X_val_fn_df.sample(5)

        # Done: display the false_negatives dataframe using metaflow.cards
        current.card.append(Table.from_dataframe(_fn))
                

if __name__ == '__main__':
    BaselineNLPFlow()

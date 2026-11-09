import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

def train():
    # 1. Load data
    df = pd.read_excel('Data/ENB2012_data.xlsx')
    
    # 2. Split data
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    y_train = df_full_train.Y1.values
    
    del df_full_train['Y1']
    del df_full_train['Y2']
    
    # 3. Vectorize features
    categorical = ['X6', 'X8']
    numerical = ['X1', 'X2', 'X3', 'X4', 'X5', 'X7']
    
    train_dict = df_full_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dict)
    
    # 4. Train Final XGBoost Model
    xgb_params = {
        'eta': 0.3,
        'max_depth': 4,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }
    
    features = dv.get_feature_names_out().tolist()
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    
    print("Training final XGBoost model...")
    model = xgb.train(xgb_params, dtrain, num_boost_round=200)
    print("Model trained successfully!")
    
    return dv, model

def save_model(dv, model, output_file):
    # The course pattern: save BOTH the vectorizer and the model together
    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
        print(f"Model and Vectorizer saved to {output_file}")

if __name__ == '__main__':
    dv, model = train()
    save_model(dv, model, 'model.bin')
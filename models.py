import warnings
warnings.filterwarnings('ignore')

#DATA MANIPULATION
import pandas as pd
from sklearn.model_selection import train_test_split

#MODELS
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#EVALUATION METRICS
from sklearn.metrics import mean_absolute_error


def decision_tree(X, y, max_leaf_nodes) -> int:
    decision_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)

    #TRAIN/TEST DATA SPLIT
    train_X, val_X, train_y, val_y = train_test_split(X, y)
    
    #FIT MODEL W/ TRAIN FEATURES
    v.fit(train_X,train_y)
    
    #PREDICT TEST FEATURES
    predicted_y = decision_model.predict(val_X)

    #EVALUATE
    # print(y.describe())
    mae = mean_absolute_error(val_y, predicted_y)
    # print(mae)
    return(mae)

def random_forest(X, y):
    forest_model = RandomForestRegressor(random_state=1)

    train_X, val_X, train_y, val_y = train_test_split(X, y)

    forest_model.fit(train_X, train_y)

    predicted_y = forest_model.predict(val_X)

    mae = mean_absolute_error(val_y, predicted_y)

    return(mae)
    



def main():
    data_path = "./housingPrices/melb_data.csv"
    df = pd.read_csv(data_path)
    df.dropna(axis=0)
    print(df.columns)
    features = ["Distance", "Rooms", "Lattitude", "Longtitude", "Bathroom", "Propertycount"]
    X = df[features]
    y = df["Price"]
    # decision_tree(X, Y)

    # compare MAE with differing values of max_leaf_nodes
    for max_leaf_nodes in [5, 50, 500, 5000]:
        my_mae = random_forest(X, y)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

if __name__ == "__main__":
    main()
    
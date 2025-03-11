import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import StandardScaler
from preprocessing import *
from modeling import *
from visualization import *

if __name__ == "__main__":
    # Load data
    # data_path = '/home/gisam1/non_imaging_data/max_data_slice.csv'
    # data = pd.read_csv(data_path)
    
    # # Filter data for the "MDD" group
    # data = data[data["Group"] == "MDD"]
    # data.to_csv('/home/gisam1/non_imaging_data/max_data_slice_mdd.csv', index = False)

    # columns_to_remove = []
    
    # # # Run the model pipeline
    # overfit_metric, model = random_forest(
    #     data,
    #     tar="DurDep",
    #     tar_skew=True,
    #     pred_skew=True,
    #     columns_to_remove=columns_to_remove,
    #     identify_predictors=True,
    #     graphs=True,
    #     dim_reduce=True
    # )

    print("Examining Titanic Dataset: ")
    data = titanic = sns.load_dataset('titanic')
    data.to_csv("../data/titanic.csv")
    # or: 
    data= pd.read_csv("../data/titanic.csv")
    print(data.columns)
    # Run the model pipeline
    overfit_metric, model = random_forest(
        data,
        tar="survived",
        tar_skew=False,
        pred_skew=False,
        columns_to_remove=["alive", "Unnamed: 0"],
        identify_predictors=True,
        graphs=True,
        dim_reduce=False,
        categorical = True
    )



    from sklearn.datasets import load_iris

    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['species'] = iris.target
    data.to_csv("../data/iris.csv")
    print(data.columns)
    print(data.shape)
    # or: 
    data=pd.read_csv("../data/iris.csv")
    

    # Run the model pipeline
    overfit_metric, model = random_forest(
        data,
        tar="species", 
        tar_skew=False,
        pred_skew=False,
        columns_to_remove=["Unnamed: 0"],
        identify_predictors=True,
        graphs=True,
        dim_reduce=False,
        categorical=True
    )

    from sklearn.datasets import load_diabetes
    import pandas as pd

    # Load Diabetes dataset
    diabetes = load_diabetes()
    data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    data['target'] = diabetes.target
    data.to_csv("../data/diabetes.csv")
    
    # or: 
    data = pd.read_csv("../data/diabetes.csv")

    # Run the model pipeline
    overfit_metric, model = random_forest(
        data,
        tar="target",
        tar_skew=True,
        pred_skew=True,
        columns_to_remove=["Unnamed: 0"],
        identify_predictors=True,
        graphs=True,
        dim_reduce=True,
        categorical = False
    )


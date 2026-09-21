import os
import mlflow
import mlflow.sklearn # Cambia según tu framework (xgboost, lightgbm, etc.)
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

from sklearn.model_selection import train_test_split

def _train_and_save_model():
    print("Iniciando entrenamiento con hiperparámetros óptimos...")
    
    # 1. Aquí cargas tus datos (desde DB, CSV local, etc.)
    # Como esto ocurre en la fase de 'build', las librerías de DB no irán a producción
    # X_train, y_train = load_data_from_db()

    # 2. Configura tu modelo con los mejores hiperparámetros
    best_params = {
        "n_estimators": 150,
        "max_depth": None,
        "random_state": 42,
        "min_samples_split": 6
    }

    DATA_PATH = "data/cars_transformed.csv"

    data = pd.read_csv(DATA_PATH)

    X = data.drop(columns=["Price"]).values
    y = data["Price"].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
    )
    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)

    artifact_path = "model"
    
    # Si la carpeta existe, mlflow fallará al intentar sobrescribirla, así que la limpiamos
    if os.path.exists(artifact_path):
        os.mkdir(artifact_path, exist_ok=True)
        
    mlflow.sklearn.save_model(
        sk_model=model, 
        path=artifact_path,
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    
    print(f"Artefacto guardado exitosamente en {artifact_path}")

if __name__ == "__main__":
    _train_and_save_model()
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import itertools
import warnings
warnings.filterwarnings('ignore')

def load_original_data():
    data = pd.read_excel('./composite_data.xlsx', sheet_name='Sheet1')

    inputs = data.iloc[:, :-2].values
    output = data.iloc[:, -2:].values

    return inputs, output, data

def load_data_split_indices(seed_dir):
    train_file = os.path.join(seed_dir, 'train_predictions.xlsx')
    test_file = os.path.join(seed_dir, 'test_predictions.xlsx')

    train_df = pd.read_excel(train_file)
    train_indices = train_df['Test_Indices'].values

    test_df = pd.read_excel(test_file)
    test_indices = test_df['Test_Indices'].values
    
    return train_indices, test_indices

def get_train_test_data(inputs, output, train_indices, test_indices):
    X_train = inputs[train_indices]
    X_test = inputs[test_indices]
    y_train = output[train_indices]
    y_test = output[test_indices]
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, target_idx=0):

    y_train_target = y_train[:, target_idx]
    y_test_target = y_test[:, target_idx]

    model.fit(X_train, y_train_target)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_r2 = r2_score(y_train_target, y_train_pred)
    train_mse = mean_squared_error(y_train_target, y_train_pred)
    train_mae = mean_absolute_error(y_train_target, y_train_pred)
    
    test_r2 = r2_score(y_test_target, y_test_pred)
    test_mse = mean_squared_error(y_test_target, y_test_pred)
    test_mae = mean_absolute_error(y_test_target, y_test_pred)
    
    return {
        'train_r2': train_r2,
        'train_mse': train_mse,
        'train_mae': train_mae,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'predictions': {
            'train_pred': y_train_pred,
            'test_pred': y_test_pred,
            'train_true': y_train_target,
            'test_true': y_test_target
        }
    }

def hyperparameter_search_rf(inputs, output, seeds_data, target_idx=0):
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [1,5,7,10, 15, 20,21],
        'min_samples_split': [2],
        'max_features': [1,5,7,10,15,20,21]
    }
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    results = []
    
    for i, params in enumerate(param_combinations):
        hyperparams = dict(zip(param_names, params))

        seed_results = []
        
        for seed, (train_indices, test_indices) in seeds_data.items():
            X_train, X_test, y_train, y_test = get_train_test_data(
                inputs, output, train_indices, test_indices
            )
            model = RandomForestRegressor(
                n_estimators=hyperparams['n_estimators'],
                max_depth=hyperparams['max_depth'],
                min_samples_split=hyperparams['min_samples_split'],
                max_features=hyperparams['max_features'],
                random_state=40,
                n_jobs=-1
            )
            result = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, target_idx)
            result['seed'] = seed
            seed_results.append(result)

        avg_test_r2 = np.mean([r['test_r2'] for r in seed_results])
        std_test_r2 = np.std([r['test_r2'] for r in seed_results])
        avg_test_mse = np.mean([r['test_mse'] for r in seed_results])
        std_test_mse = np.std([r['test_mse'] for r in seed_results])

        results.append({
            'hyperparameters': hyperparams,
            'avg_test_r2': avg_test_r2,
            'std_test_r2': std_test_r2,
            'avg_test_mse': avg_test_mse,
            'std_test_mse': std_test_mse,
            'seed_results': seed_results
        })
    
    return results

def hyperparameter_search_gb(inputs, output, seeds_data, target_idx=0):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.0001, 0.05, 0.1, 0.2,0.5],
        'max_depth': [5,7,10,15,20],
        'min_samples_split': [2],
        'max_features': [5,7,10,15,20]
    }

    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    results = []
    
    for i, params in enumerate(param_combinations):
        hyperparams = dict(zip(param_names, params))

        seed_results = []
        
        for seed, (train_indices, test_indices) in seeds_data.items():
            X_train, X_test, y_train, y_test = get_train_test_data(
                inputs, output, train_indices, test_indices
            )
            model = GradientBoostingRegressor(
                n_estimators=hyperparams['n_estimators'],
                learning_rate=hyperparams['learning_rate'],
                max_depth=hyperparams['max_depth'],
                min_samples_split=hyperparams['min_samples_split'],
                max_features=hyperparams['max_features'],
                random_state=40
            )
            result = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, target_idx)
            result['seed'] = seed
            seed_results.append(result)

        avg_test_r2 = np.mean([r['test_r2'] for r in seed_results])
        std_test_r2 = np.std([r['test_r2'] for r in seed_results])
        avg_test_mse = np.mean([r['test_mse'] for r in seed_results])
        std_test_mse = np.std([r['test_mse'] for r in seed_results])

        results.append({
            'hyperparameters': hyperparams,
            'avg_test_r2': avg_test_r2,
            'std_test_r2': std_test_r2,
            'avg_test_mse': avg_test_mse,
            'std_test_mse': std_test_mse,
            'seed_results': seed_results
        })
    return results

def hyperparameter_search_svm(inputs, output, seeds_data, target_idx):
    param_grid = {
        'C': [0.1, 1, 10, 50],
        'epsilon': [0.001, 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }

    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    results = []
    
    for i, params in enumerate(param_combinations):
        hyperparams = dict(zip(param_names, params))

        seed_results = []
        
        for seed, (train_indices, test_indices) in seeds_data.items():
            X_train, X_test, y_train, y_test = get_train_test_data(
                inputs, output, train_indices, test_indices
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = SVR(
                C=hyperparams['C'],
                epsilon=hyperparams['epsilon'],
                kernel=hyperparams['kernel'],
                max_iter=1000
            )

            result = train_and_evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, target_idx)
            result['seed'] = seed
            seed_results.append(result)

        avg_test_r2 = np.mean([r['test_r2'] for r in seed_results])
        std_test_r2 = np.std([r['test_r2'] for r in seed_results])
        avg_test_mse = np.mean([r['test_mse'] for r in seed_results])
        std_test_mse = np.std([r['test_mse'] for r in seed_results])
        
        # 存储结果
        results.append({
            'hyperparameters': hyperparams,
            'avg_test_r2': avg_test_r2,
            'std_test_r2': std_test_r2,
            'avg_test_mse': avg_test_mse,
            'std_test_mse': std_test_mse,
            'seed_results': seed_results
        })
    return results

def save_results(results, algorithm_name, target_idx, save_dir):

    results_dir = os.path.join(save_dir, f'{algorithm_name}_results')
    os.makedirs(results_dir, exist_ok=True)
    results_data = []
    for result in results:
        row = {
            'algorithm': algorithm_name,
            'target': target_idx + 1,
            'avg_test_r2': result['avg_test_r2'],
            'std_test_r2': result['std_test_r2'],
            'avg_test_mse': result['avg_test_mse'],
            'std_test_mse': result['std_test_mse']
        }
        row.update(result['hyperparameters'])
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)

    results_df = results_df.sort_values('avg_test_r2', ascending=False)

    results_file = os.path.join(results_dir, f'{algorithm_name}_target{target_idx+1}_results.csv')
    results_df.to_csv(results_file, index=False)
    return results_df

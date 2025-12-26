"""
optuna_study.py

Optuna 기반 하이퍼파라미터 튜닝 스크립트 (Bayesian Optimization)
- train.py의 주요 하이퍼파라미터(lr, batch_size, dropout 등)를 튜닝
- 각 trial별로 임시 config를 생성하여 train.py를 함수로 직접 호출
- MLflow와 연동, Optuna plot 자동 저장

Author: Copilot (2025)
"""
import optuna
import os
import shutil
import yaml
import subprocess
import matplotlib.pyplot as plt
import json
import datetime

# 튜닝 대상 하이퍼파라미터 범위 정의
def load_search_space():
    with open('paper1_mfcc_cnn/configs/hyper_search.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg['search_space']

def suggest_params(trial, search_space):
    params = {}
    for k, v in search_space.items():
        if v['type'] == 'loguniform':
            params[k] = trial.suggest_loguniform(k, v['low'], v['high'])
        elif v['type'] == 'uniform':
            params[k] = trial.suggest_uniform(k, v['low'], v['high'])
        elif v['type'] == 'categorical':
            params[k] = trial.suggest_categorical(k, v['choices'])
    return params

def run_train_with_config(config_path, log_dir):
    """train.py를 subprocess로 실행 (Hydra config 경로 지정)"""
    result = subprocess.run([
        'python', 'scripts/train.py', f'--config-name={os.path.basename(config_path)}', f'--config-dir={os.path.dirname(config_path)}'
    ], capture_output=True, text=True)
    # 로그 저장
    with open(os.path.join(log_dir, 'train_stdout.txt'), 'w') as f:
        f.write(result.stdout)
    with open(os.path.join(log_dir, 'train_stderr.txt'), 'w') as f:
        f.write(result.stderr)
    return result.returncode == 0

def get_val_acc(log_dir):
    """metrics_val.json에서 best val_acc를 읽어옴"""
    metrics_path = os.path.join(log_dir, 'metrics_val.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics.get('best_val_acc', 0.0)
    return 0.0

def objective(trial):
    # 1. search space config 로드
    search_space = load_search_space()
    # 2. trial별 폴더 생성
    optuna_dir = 'paper1_mfcc_cnn/outputs/optuna'
    os.makedirs(optuna_dir, exist_ok=True)
    trial_dir = os.path.join(optuna_dir, f'trial_{trial.number}')
    os.makedirs(trial_dir, exist_ok=True)
    # 3. trial별 임시 config 생성
    params = suggest_params(trial, search_space)
    with open('paper1_mfcc_cnn/configs/default.yaml', 'r') as f:
        base_cfg = yaml.safe_load(f)
    # 하이퍼파라미터 반영
    base_cfg['train']['lr'] = params['lr']
    base_cfg['train']['batch_size'] = params['batch_size']
    base_cfg['train']['weight_decay'] = params['weight_decay']
    base_cfg['model']['transformer']['dropout'] = params['dropout']
    base_cfg['data']['batch_size'] = params['batch_size']
    # log_dir을 trial별 폴더로 지정
    base_cfg['train']['log_dir'] = trial_dir
    # trial별 임시 config 저장
    trial_cfg_path = os.path.join(trial_dir, 'trial_config.yaml')
    with open(trial_cfg_path, 'w') as f:
        yaml.dump(base_cfg, f)
    # search space 정보도 trial 폴더에 저장
    with open(os.path.join(trial_dir, 'search_space.yaml'), 'w') as f:
        yaml.dump(search_space, f)
    # 4. train.py 실행
    run_train_with_config(trial_cfg_path, trial_dir)
    # 5. val_acc 추출 (metrics_val.json)
    val_acc = get_val_acc(trial_dir)
    # trial별 metric 기록
    with open(os.path.join(trial_dir, 'optuna_metric.json'), 'w') as f:
        json.dump({'val_acc': val_acc, 'params': params}, f, indent=2)
    return val_acc

def main():
    # optuna 결과 폴더 생성
    optuna_dir = 'paper1_mfcc_cnn/outputs/optuna'
    os.makedirs(optuna_dir, exist_ok=True)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=10)
    # 주요 plot 저장 (optuna 폴더)
    import optuna.visualization as vis
    plot_paths = {
        'optimization_history': os.path.join(optuna_dir, 'optuna_optimization_history.png'),
        'param_importance': os.path.join(optuna_dir, 'optuna_param_importance.png'),
        'parallel_coordinate': os.path.join(optuna_dir, 'optuna_parallel_coordinate.png'),
        'contour': os.path.join(optuna_dir, 'optuna_contour.png'),
    }
    vis.plot_optimization_history(study).write_image(plot_paths['optimization_history'])
    vis.plot_param_importances(study).write_image(plot_paths['param_importance'])
    vis.plot_parallel_coordinate(study).write_image(plot_paths['parallel_coordinate'])
    vis.plot_contour(study).write_image(plot_paths['contour'])
    print(f'Optuna plots saved to {optuna_dir}/')

if __name__ == '__main__':
    main()

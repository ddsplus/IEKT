# IEKT: 8/2 Data Processing and Training

This repo trains IEKT with a strict `80% train / 20% test` split (no validation set).

## 1. Requirements

- Python 3.8+
- `torch`
- `pandas`
- `numpy`
- `tqdm`
- `scikit-learn`

Install:

```bash
pip install torch pandas numpy tqdm scikit-learn
```

## 2. Raw Data Paths

Place raw CSV files at:

- `Data/ASSIST2017/anonymized_full_release_competition_dataset.csv`
- `Data/Statics2011/AllData_student_step_2011F.csv`

## 3. Preprocess (8/2, No Valid)

Preprocessing outputs:

- `history_train.pkl`
- `history_test.pkl`
- `problem_skill_maxSkillOfProblem_number.pkl`

### 3.1 ASSIST2017

```bash
python Data/preprocess_iekt_assist2017.py \
  --input_csv Data/ASSIST2017/anonymized_full_release_competition_dataset.csv \
  --output_dir Data/iekt_assist2017 \
  --min_seq_len 10 \
  --max_concepts 5 \
  --train_ratio 0.8 \
  --seed 42
```

### 3.2 Statics2011

```bash
python Data/preprocess_iekt_statics2011.py \
  --input_csv Data/Statics2011/AllData_student_step_2011F.csv \
  --output_dir Data/iekt_statics2011 \
  --min_seq_len 10 \
  --max_concepts 5 \
  --train_ratio 0.8 \
  --seed 42
```

## 4. Remove Old Cached Dataset Objects

If you previously created `dataset_valid.pkl` or old cached datasets, remove them before retraining:

```bash
rm -f Data/iekt_assist2017/dataset_*.pkl
rm -f Data/iekt_statics2011/dataset_*.pkl
```

## 5. Train

`main.py` now loads only `train` and `test`.

### 5.1 ASSIST2017

```bash
python main.py \
  --run_dir runs/assist2017 \
  --data_dir Data/iekt_assist2017/ \
  --model iekt \
  --device 0 \
  --n_epochs 300 \
  --batch_size 32 \
  --lr 1e-3 \
  --seq_len 200
```

### 5.2 Statics2011

```bash
python main.py \
  --run_dir runs/statics2011 \
  --data_dir Data/iekt_statics2011/ \
  --model iekt \
  --device 0 \
  --n_epochs 300 \
  --batch_size 32 \
  --lr 1e-3 \
  --seq_len 200
```

CPU example:

```bash
python main.py --run_dir runs/cpu --data_dir Data/iekt_assist2017/ --device -1
```

## 6. Per-Epoch Output

Each epoch logs:

- `test_auc`
- `test_acc`
- `loss`

Example:

```text
Epoch: 010, Loss: 0.1234567, test_auc: 0.8123456, test_acc: 0.7456789
```

Log file:

- `<run_dir>/log.txt`


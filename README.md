# IEKT: 数据预处理与训练指南

这个仓库是 IEKT（Tracing Knowledge State with Individual Cognition and Acquisition Estimation）的实现。  
当前已补充 `ASSIST2017` 和 `Statics2011` 两个数据集的 IEKT 预处理脚本，并且训练脚本每个 epoch 会输出测试集 `AUC` 和 `ACC`。

## 1. 环境准备

建议 Python 3.8+。

最少依赖：
- `torch`
- `pandas`
- `numpy`
- `tqdm`
- `scikit-learn`

示例安装：

```bash
pip install torch pandas numpy tqdm scikit-learn
```

## 2. 原始数据位置

将原始 CSV 放在以下位置（默认路径）：

- `Data/ASSIST2017/anonymized_full_release_competition_dataset.csv`
- `Data/Statics2011/AllData_student_step_2011F.csv`

## 3. 预处理数据（生成 IEKT 可读 pkl）

IEKT 训练依赖以下文件：
- `history_train.pkl`
- `history_valid.pkl`
- `history_test.pkl`
- `problem_skill_maxSkillOfProblem_number.pkl`

### 3.1 处理 ASSIST2017

运行：

```bash
python Data/preprocess_iekt_assist2017.py \
  --input_csv Data/ASSIST2017/anonymized_full_release_competition_dataset.csv \
  --output_dir Data/iekt_assist2017 \
  --min_seq_len 10 \
  --max_concepts 5 \
  --train_ratio 0.8 \
  --valid_ratio 0.1 \
  --seed 42
```

输出目录：`Data/iekt_assist2017/`

### 3.2 处理 Statics2011

运行：

```bash
python Data/preprocess_iekt_statics2011.py \
  --input_csv Data/Statics2011/AllData_student_step_2011F.csv \
  --output_dir Data/iekt_statics2011 \
  --min_seq_len 10 \
  --max_concepts 5 \
  --train_ratio 0.8 \
  --valid_ratio 0.1 \
  --seed 42
```

输出目录：`Data/iekt_statics2011/`

## 4. 训练

`main.py` 通过 `--data_dir` 读取预处理后的目录。注意目录末尾建议带 `/`。

### 4.1 在 ASSIST2017 上训练

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

### 4.2 在 Statics2011 上训练

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

如果只用 CPU：

```bash
python main.py --run_dir runs/cpu --data_dir Data/iekt_assist2017/ --device -1
```

## 5. 每轮输出指标

训练日志每个 epoch 会输出：
- `test_auc`
- `test_acc`
- `valid_auc`
- `valid_acc`
- `loss`

日志示例（格式）：

```text
Epoch: 010, Loss: 0.1234567, test_auc: 0.8123456, test_acc: 0.7456789, valid_auc: 0.8012345, valid_acc: 0.7345678
```

同时日志会写入：`<run_dir>/log.txt`。

## 6. 常用参数说明

- `--data_dir`：预处理输出目录（必须包含 `history_*.pkl` 和 `problem_skill_maxSkillOfProblem_number.pkl`）
- `--run_dir`：训练日志和模型保存目录
- `--device`：`-1` 为 CPU，`0/1/...` 为 GPU 编号
- `--n_epochs`：训练轮数
- `--batch_size`：批大小
- `--seq_len`：序列截断长度
- `--save_every`：每多少个 epoch 保存一次模型
- `--checkpoint_path`：加载已有模型继续训练（默认 `none`）

## 7. 快速自检

如果训练前想确认数据可读，检查以下文件是否存在：

- `Data/iekt_assist2017/history_train.pkl`
- `Data/iekt_assist2017/history_valid.pkl`
- `Data/iekt_assist2017/history_test.pkl`
- `Data/iekt_assist2017/problem_skill_maxSkillOfProblem_number.pkl`

或者对应的 `Data/iekt_statics2011/` 目录。

---

论文信息可参考仓库中的 PDF：  
`Tracing Knowledge State with Individual Cognition and Acquisition Estimation.pdf`

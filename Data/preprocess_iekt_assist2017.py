#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess ASSIST2017 for IEKT.

Outputs:
  - history_train.pkl
  - history_valid.pkl
  - history_test.pkl
  - problem_skill_maxSkillOfProblem_number.pkl
"""

import argparse
import os
import pickle
import random
from collections import defaultdict

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess ASSIST2017 for IEKT")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="Data/ASSIST2017/anonymized_full_release_competition_dataset.csv",
    )
    parser.add_argument("--output_dir", type=str, default="Data/iekt_assist2017/")
    parser.add_argument("--min_seq_len", type=int, default=10)
    parser.add_argument("--max_concepts", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_df(path):
    df = pd.read_csv(path, encoding="utf-8")
    required = ["studentId", "problemId", "skill", "correct", "startTime"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["studentId", "problemId", "skill", "correct"]).copy()
    df["studentId"] = df["studentId"].astype(str).str.strip()
    df["problemId"] = df["problemId"].astype(str).str.strip()
    df["skill"] = df["skill"].astype(str).str.strip()
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df[df["correct"].isin([0, 1])].copy()
    df["correct"] = df["correct"].astype(int)
    df["startTime"] = pd.to_numeric(df["startTime"], errors="coerce")
    df = df.sort_values(["studentId", "startTime"], kind="mergesort").reset_index(drop=True)
    return df


def build_mappings(df):
    problem_values = sorted(df["problemId"].unique().tolist())
    skill_values = sorted(df["skill"].unique().tolist())

    problem_to_id = {raw: i + 1 for i, raw in enumerate(problem_values)}
    skill_to_id = {raw: i + 1 for i, raw in enumerate(skill_values)}
    return problem_to_id, skill_to_id


def group_user_records(df, problem_to_id, skill_to_id):
    grouped = defaultdict(list)
    for row in df.itertuples(index=False):
        uid = str(row.studentId)
        pid = problem_to_id[row.problemId]
        sid = skill_to_id[row.skill]
        correct = int(row.correct)
        grouped[uid].append((pid, [sid], correct))
    return grouped


def split_users(user_ids, train_ratio, valid_ratio, seed):
    if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("Require 0 < train_ratio, valid_ratio and train_ratio + valid_ratio < 1")
    ids = list(user_ids)
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    n_total = len(ids)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)

    train_ids = ids[:n_train]
    valid_ids = ids[n_train : n_train + n_valid]
    test_ids = ids[n_train + n_valid :]
    return train_ids, valid_ids, test_ids


def build_histories(user_ids, grouped, min_seq_len, max_concepts):
    histories = {}
    for uid in user_ids:
        interactions = grouped[uid]
        if len(interactions) < min_seq_len:
            continue

        records = []
        for order_id, (pid, skills, response) in enumerate(interactions):
            pad_skills = skills[:max_concepts] + [0] * max(0, max_concepts - len(skills))
            records.append((order_id, int(pid), pad_skills, int(response)))
        histories[uid] = [len(records), records]
    return histories


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_df(args.input_csv)
    problem_to_id, skill_to_id = build_mappings(df)
    grouped = group_user_records(df, problem_to_id, skill_to_id)
    user_ids = list(grouped.keys())

    train_ids, valid_ids, test_ids = split_users(
        user_ids, args.train_ratio, args.valid_ratio, args.seed
    )

    train_hist = build_histories(train_ids, grouped, args.min_seq_len, args.max_concepts)
    valid_hist = build_histories(valid_ids, grouped, args.min_seq_len, args.max_concepts)
    test_hist = build_histories(test_ids, grouped, args.min_seq_len, args.max_concepts)

    with open(os.path.join(args.output_dir, "history_train.pkl"), "wb") as f:
        pickle.dump(train_hist, f)
    with open(os.path.join(args.output_dir, "history_valid.pkl"), "wb") as f:
        pickle.dump(valid_hist, f)
    with open(os.path.join(args.output_dir, "history_test.pkl"), "wb") as f:
        pickle.dump(test_hist, f)

    problem_number = len(problem_to_id) + 1
    concept_number = len(skill_to_id) + 1
    meta = (problem_number, concept_number, args.max_concepts)
    with open(
        os.path.join(args.output_dir, "problem_skill_maxSkillOfProblem_number.pkl"), "wb"
    ) as f:
        pickle.dump(meta, f)

    print("ASSIST2017 preprocess done")
    print(f"output_dir: {args.output_dir}")
    print(f"train users kept: {len(train_hist)}")
    print(f"valid users kept: {len(valid_hist)}")
    print(f"test users kept: {len(test_hist)}")
    print(f"problem_number(with pad): {problem_number}")
    print(f"concept_number(with pad): {concept_number}")
    print(f"max_concepts: {args.max_concepts}")


if __name__ == "__main__":
    main()

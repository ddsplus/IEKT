#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess Statics2011 for IEKT.

Outputs:
  - history_train.pkl
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
    parser = argparse.ArgumentParser(description="Preprocess Statics2011 for IEKT")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="Data/Statics2011/AllData_student_step_2011F.csv",
    )
    parser.add_argument("--output_dir", type=str, default="Data/iekt_statics2011/")
    parser.add_argument("--min_seq_len", type=int, default=10)
    parser.add_argument("--max_concepts", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_kcs(kc_text):
    value = str(kc_text).strip()
    if not value or value == "." or value.lower() == "nan":
        return []
    return [x.strip() for x in value.split("~~") if x.strip() and x.strip() != "."]


def load_df(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = [
        "Anon Student Id",
        "Problem Name",
        "Step Name",
        "First Transaction Time",
        "First Attempt",
        "KC (F2011)",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["Anon Student Id", "Problem Name", "Step Name", "KC (F2011)"]).copy()
    df["Anon Student Id"] = df["Anon Student Id"].astype(str).str.strip()
    df["Problem Name"] = df["Problem Name"].astype(str).str.strip()
    df["Step Name"] = df["Step Name"].astype(str).str.strip()

    df["problem_key"] = df["Problem Name"] + "::" + df["Step Name"]
    df["skills"] = df["KC (F2011)"].apply(split_kcs)
    df = df[df["skills"].map(len) > 0].copy()

    attempt = df["First Attempt"].astype(str).str.strip().str.lower()
    df["correct"] = (attempt == "correct").astype(int)
    df["ts"] = pd.to_datetime(df["First Transaction Time"], errors="coerce")
    df = df.sort_values(["Anon Student Id", "ts"], kind="mergesort").reset_index(drop=True)
    return df


def build_mappings(df):
    problem_values = sorted(df["problem_key"].unique().tolist())
    all_skills = set()
    for values in df["skills"].tolist():
        all_skills.update(values)
    skill_values = sorted(all_skills)

    problem_to_id = {raw: i + 1 for i, raw in enumerate(problem_values)}
    skill_to_id = {raw: i + 1 for i, raw in enumerate(skill_values)}
    return problem_to_id, skill_to_id


def group_user_records(df, problem_to_id, skill_to_id, max_concepts):
    grouped = defaultdict(list)
    for uid, problem_key, skills, correct in df[
        ["Anon Student Id", "problem_key", "skills", "correct"]
    ].itertuples(index=False, name=None):
        uid = str(uid)
        pid = problem_to_id[problem_key]
        skill_ids = [skill_to_id[s] for s in skills][:max_concepts]
        grouped[uid].append((pid, skill_ids, correct))
    return grouped


def split_users(user_ids, train_ratio, seed):
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("Require 0 < train_ratio < 1")
    ids = list(user_ids)
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    n_total = len(ids)
    n_train = int(n_total * train_ratio)

    train_ids = ids[:n_train]
    test_ids = ids[n_train:]
    return train_ids, test_ids


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
    grouped = group_user_records(df, problem_to_id, skill_to_id, args.max_concepts)
    user_ids = list(grouped.keys())

    train_ids, test_ids = split_users(user_ids, args.train_ratio, args.seed)

    train_hist = build_histories(train_ids, grouped, args.min_seq_len, args.max_concepts)
    test_hist = build_histories(test_ids, grouped, args.min_seq_len, args.max_concepts)

    with open(os.path.join(args.output_dir, "history_train.pkl"), "wb") as f:
        pickle.dump(train_hist, f)
    with open(os.path.join(args.output_dir, "history_test.pkl"), "wb") as f:
        pickle.dump(test_hist, f)

    problem_number = len(problem_to_id) + 1
    concept_number = len(skill_to_id) + 1
    meta = (problem_number, concept_number, args.max_concepts)
    with open(
        os.path.join(args.output_dir, "problem_skill_maxSkillOfProblem_number.pkl"), "wb"
    ) as f:
        pickle.dump(meta, f)

    print("Statics2011 preprocess done")
    print(f"output_dir: {args.output_dir}")
    print(f"train users kept: {len(train_hist)}")
    print(f"test users kept: {len(test_hist)}")
    print(f"problem_number(with pad): {problem_number}")
    print(f"concept_number(with pad): {concept_number}")
    print(f"max_concepts: {args.max_concepts}")


if __name__ == "__main__":
    main()

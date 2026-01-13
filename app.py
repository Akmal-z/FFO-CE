# ============================================================
# Employee Shift Scheduling (FFO – Firefly Optimization)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt

# ============================================================
# CONFIG (UNCHANGED)
# ============================================================

st.title(" Employee Shift Scheduling (FFO)✨ ")

n_departments = 6
n_days = 7
n_periods = 28
SHIFT_LENGTH = 14

PENALTY_SHORTAGE = 200
PENALTY_OVERHOURS = 150
PENALTY_DAYS_MIN = 300
PENALTY_SHIFT_BREAK = 100
PENALTY_NONCONSEC = 200

# ============================================================
# LOAD DEMAND (UNCHANGED)
# ============================================================

DEMAND = np.zeros((n_departments, n_days, n_periods), dtype=int)
folder_path = "./Demand/"

for dept in range(n_departments):
    file_path = os.path.join(folder_path, f"Dept{dept+1}.xlsx")
    if not os.path.exists(file_path):
        st.sidebar.error(f"❌ Dept{dept+1}.xlsx not found")
        continue
    df = pd.read_excel(file_path, header=None)
    df = df.iloc[1:1+n_days, 1:1+n_periods]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    DEMAND[dept] = df.values

# ============================================================
# HELPER FUNCTIONS (UNCHANGED)
# ============================================================

def longest_consecutive_ones(arr):
    max_len = curr = 0
    for v in arr:
        if v == 1:
            curr += 1
            max_len = max(max_len, curr)
        else:
            curr = 0
    return max_len


def pareto_filter(points):
    pareto = []
    for p in points:
        if not any((q[0] <= p[0] and q[1] <= p[1]) and q != p for q in points):
            pareto.append(p)
    return pareto


def compute_penalty_breakdown(schedule, demand, max_hours):
    total_shortage = total_overwork = total_days_min = 0
    total_shift_break = total_nonconsec = 0

    for dept in range(n_departments):
        for d in range(n_days):
            for t in range(n_periods):
                assigned = np.sum(schedule[dept, d, t])
                required = demand[dept, d, t]
                if assigned < required:
                    total_shortage += (required - assigned) * PENALTY_SHORTAGE

        for e in range(schedule.shape[3]):
            total_hours = np.sum(schedule[:, :, :, e])
            if total_hours > max_hours:
                total_overwork += (total_hours - max_hours) * PENALTY_OVERHOURS

            days_worked = np.sum(np.sum(schedule[:, :, :, e], axis=2) > 0)
            if days_worked < (n_days - 1):
                total_days_min += PENALTY_DAYS_MIN

        for d in range(n_days):
            for e in range(schedule.shape[3]):
                daily = schedule[dept, d, :, e]
                worked = np.sum(daily)
                if worked > 0 and worked != SHIFT_LENGTH:
                    total_shift_break += PENALTY_SHIFT_BREAK
                if worked == SHIFT_LENGTH and longest_consecutive_ones(daily) < SHIFT_LENGTH:
                    total_nonconsec += PENALTY_NONCONSEC

    return {
        "total_fitness": total_shortage + total_overwork +
                         total_days_min + total_shift_break + total_nonconsec,
        "shortage": total_shortage,
        "overwork": total_overwork,
        "days_min": total_days_min,
        "shift_break": total_shift_break,
        "nonconsec": total_nonconsec
    }


def compute_objectives(schedule, demand, max_hours):
    total_shortage = workload_penalty = 0
    for dept in range(n_departments):
        for d in range(n_days):
            for t in range(n_periods):
                total_shortage += max(demand[dept, d, t] - np.sum(schedule[dept, d, t]), 0)
        for e in range(schedule.shape[3]):
            hours = np.sum(schedule[:, :, :, e])
            if hours > max_hours:
                workload_penalty += (hours - max_hours)
    return total_shortage, workload_penalty


def fitness(schedule, demand, max_hours):
    return compute_penalty_breakdown(schedule, demand, max_hours)["total_fitness"]


def generate_min_one_off_schedule(n_employees, n_days):
    off = np.zeros((n_employees, n_days), dtype=int)
    for e in range(n_employees):
        off[e, random.randint(0, n_days-1)] = 1
    return off

# ============================================================
# 🔥 TRUE FFO (SAME INTERFACE AS ACO)
# ============================================================

def ACO_scheduler(demand, n_employees_per_dept, n_ants, n_iter,
                  alpha, evaporation, Q, max_hours, early_stop):

    population = []
    fitness_vals = []
    fitness_history = []
    pareto_raw = []
    pareto_schedules = []

    best_score_global = float("inf")
    best_schedule_global = None
    best_off_schedules_global = None
    no_improve = 0
    start_time = time.time()

    max_emp = max(n_employees_per_dept)

    # -------- Initialize Fireflies --------
    for _ in range(n_ants):
        schedule = np.zeros((n_departments, n_days, n_periods, max_emp))
        off_schedules = []

        for dept in range(n_departments):
            n_emp = n_employees_per_dept[dept]
            off = generate_min_one_off_schedule(n_emp, n_days)
            off_schedules.append(off)

            for d in range(n_days):
                for e in range(n_emp):
                    if off[e, d] == 1 or random.random() < REST_PROB:
                        continue
                    if random.random() < 0.5:
                        schedule[dept, d, 0:SHIFT_LENGTH, e] = 1
                    else:
                        schedule[dept, d, 14:14+SHIFT_LENGTH, e] = 1

        population.append(schedule)
        fitness_vals.append(fitness(schedule, demand, max_hours))

    # -------- FFO Iterations --------
    for it in range(n_iter):
        all_scores_iter = []

        for i in range(n_ants):
            for j in range(n_ants):
                if fitness_vals[j] < fitness_vals[i]:
                    r = np.linalg.norm(population[i] - population[j])
                    beta = np.exp(-r * r)

                    new = population[i] + beta * (population[j] - population[i])
                    new = np.clip(new, 0, 1)

                    # probabilistic binary update
                    new = (np.random.rand(*new.shape) < new).astype(int)

                    # mutation (VERY IMPORTANT)
                    mutation = np.random.rand(*new.shape) < 0.01
                    new[mutation] = 1 - new[mutation]

                    score = fitness(new, demand, max_hours)
                    if score < fitness_vals[i]:
                        population[i] = new
                        fitness_vals[i] = score

            pareto_raw.append(compute_objectives(population[i], demand, max_hours))
            pareto_schedules.append(population[i].copy())
            all_scores_iter.append(fitness_vals[i])

        iteration_best_score = min(fitness_vals)
        iteration_best_schedule = population[np.argmin(fitness_vals)]

        if iteration_best_score < best_score_global:
            best_score_global = iteration_best_score
            best_schedule_global = iteration_best_schedule.copy()
            best_off_schedules_global = off_schedules.copy()
            no_improve = 0
        else:
            no_improve += 1

        fitness_history.append({
            "iteration": it + 1,
            "best": iteration_best_score,
            "mean": np.mean(all_scores_iter),
            "worst": np.max(all_scores_iter)
        })

        if no_improve >= early_stop:
            break

    pareto_filtered = pareto_filter(pareto_raw)
    filtered_schedules = [
        pareto_schedules[i] for i, p in enumerate(pareto_raw)
        if p in pareto_filtered
    ]

    best_score_from_pareto = float("inf")
    best_schedule_final = None
    best_index = None

    for idx, sched in enumerate(filtered_schedules):
        score = fitness(sched, demand, max_hours)
        if score < best_score_from_pareto:
            best_score_from_pareto = score
            best_schedule_final = sched.copy()
            best_index = idx

    run_time = time.time() - start_time

    return best_schedule_final, best_score_from_pareto, fitness_history, pareto_filtered, run_time, best_off_schedules_global, best_index

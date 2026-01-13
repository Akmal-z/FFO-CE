# ============================================================
# Employee Shift Scheduling using Firefly Optimization (FFO)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt

# ============================================================
# 🔴 PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ============================================================
st.set_page_config(
    page_title="Employee Shift Scheduling (FFO)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# TITLE
# ============================================================
st.title("Employee Shift Scheduling (FFO) ✨")

# ============================================================
# PROBLEM CONFIG (UNCHANGED)
# ============================================================
n_departments = 6
n_days = 7
n_periods = 28
SHIFT_LENGTH = 14

# ============================================================
# PENALTIES (UNCHANGED)
# ============================================================
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
            hours = np.sum(schedule[:, :, :, e])
            if hours > max_hours:
                total_overwork += (hours - max_hours) * PENALTY_OVERHOURS

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
    shortage = workload = 0
    for dept in range(n_departments):
        for d in range(n_days):
            for t in range(n_periods):
                shortage += max(demand[dept, d, t] - np.sum(schedule[dept, d, t]), 0)
        for e in range(schedule.shape[3]):
            hours = np.sum(schedule[:, :, :, e])
            if hours > max_hours:
                workload += (hours - max_hours)
    return shortage, workload


def fitness(schedule, demand, max_hours):
    return compute_penalty_breakdown(schedule, demand, max_hours)["total_fitness"]


def generate_min_one_off_schedule(n_employees, n_days):
    off = np.zeros((n_employees, n_days), dtype=int)
    for e in range(n_employees):
        off[e, random.randint(0, n_days - 1)] = 1
    return off

# ============================================================
# 🔥 FIREfly OPTIMIZATION ALGORITHM (FFO)
# ============================================================
def FFO_scheduler(demand, n_employees_per_dept, n_fireflies, n_iter,
                  alpha, beta0, gamma, max_hours, early_stop):

    max_emp = max(n_employees_per_dept)
    population = []
    fitness_vals = []
    pareto_raw = []
    pareto_schedules = []
    fitness_history = []

    best_score = float("inf")
    best_schedule = None
    best_off_schedules = None
    no_improve = 0

    start_time = time.time()

    # -------- Initialize fireflies --------
    for _ in range(n_fireflies):
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

    # -------- Main FFO loop --------
    for it in range(n_iter):
        all_scores = []

        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness_vals[j] < fitness_vals[i]:
                    r = np.linalg.norm(population[i] - population[j])
                    beta = beta0 * np.exp(-gamma * r * r)

                    new = population[i] + beta * (population[j] - population[i])
                    new = np.clip(new, 0, 1)

                    # binary probabilistic update
                    new = (np.random.rand(*new.shape) < new).astype(int)

                    # mutation (prevents flat line)
                    mutation = np.random.rand(*new.shape) < 0.01
                    new[mutation] = 1 - new[mutation]

                    score = fitness(new, demand, max_hours)
                    if score < fitness_vals[i]:
                        population[i] = new
                        fitness_vals[i] = score

            s, w = compute_objectives(population[i], demand, max_hours)
            pareto_raw.append((s, w))
            pareto_schedules.append(population[i].copy())
            all_scores.append(fitness_vals[i])

        iter_best = min(fitness_vals)
        fitness_history.append({
            "iteration": it + 1,
            "best": iter_best,
            "mean": np.mean(all_scores),
            "worst": np.max(all_scores)
        })

        if iter_best < best_score:
            best_score = iter_best
            best_schedule = population[np.argmin(fitness_vals)].copy()
            best_off_schedules = off_schedules.copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stop:
            break

    pareto_filtered = pareto_filter(pareto_raw)

    run_time = time.time() - start_time
    return best_schedule, best_score, fitness_history, pareto_filtered, run_time, best_off_schedules

# ============================================================
# SIDEBAR — FFO PARAMETERS (VALUES UNCHANGED)
# ============================================================
st.sidebar.header("FFO Parameters")

n_fireflies = st.sidebar.slider("Fireflies", 5, 50, 20)
n_iter = st.sidebar.slider("Iterations", 10, 500, 50)
early_stop = st.sidebar.slider("Early Stop Iterations", 1, 50, 10)

alpha = st.sidebar.slider("Alpha", 0.1, 5.0, 1.0)
beta0 = st.sidebar.slider("Beta₀", 0.1, 2.0, 1.0)
gamma = st.sidebar.slider("Gamma", 0.01, 1.0, 0.1)

REST_PROB = st.sidebar.slider("Rest Probability (REST_PROB)", 0.0, 0.8, 0.35, step=0.05)
max_hours = st.sidebar.slider("Max Hours / Week", 20, 60, 40)

st.sidebar.header("Employees per Department")
n_employees_per_dept = [
    st.sidebar.number_input(f"Dept {i+1} Employees", 1, 50, 20)
    for i in range(n_departments)
]

# ============================================================
# RUN FFO
# ============================================================
if st.sidebar.button("Run FFO"):
    best_schedule, best_score, fitness_history, pareto_data, run_time, best_off_schedules = \
        FFO_scheduler(DEMAND, n_employees_per_dept, n_fireflies, n_iter,
                      alpha, beta0, gamma, max_hours, early_stop)

    st.success(f"Best Fitness Score: {best_score:.2f}")
    st.info(f"Computation Time: {run_time:.2f} seconds")

    # ================= Fitness Convergence =================
    iters = [x["iteration"] for x in fitness_history]
    best_vals = [x["best"] for x in fitness_history]

    fig, ax = plt.subplots()
    ax.plot(iters, best_vals, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness Convergence (FFO)")
    st.pyplot(fig)

    # ================= Fitness Breakdown =================
    st.subheader("Fitness Breakdown")
    breakdown = compute_penalty_breakdown(best_schedule, DEMAND, max_hours)
    st.json(breakdown)

    # ================= Radar Chart (UNCHANGED STYLE) =================
    st.subheader("🎯 Constraint Balance (Radar Chart)")
    bd = breakdown

    cats = ['Shortage', 'Overwork', 'Min Days', 'Shift Break', 'Consecutive']
    vals = [bd['shortage'], bd['overwork'], bd['days_min'], bd['shift_break'], bd['nonconsec']]

    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals += vals[:1]

    fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax_radar.plot(angles, vals, linewidth=2, linestyle='solid', color='#E63946')
    ax_radar.fill(angles, vals, '#E63946', alpha=0.25)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(cats)
    ax_radar.set_title("Penalty Distribution (Smaller shape is better)")
    st.pyplot(fig_radar)

# ============================================================
# Employee Shift Scheduling using Firefly Optimization (FFO)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(page_title="Employee Shift Scheduling (FFO)", layout="wide")
st.title("🔥 Employee Shift Scheduling using Firefly Optimization (FFO)")

# ============================================================
# PROBLEM SETUP
# ============================================================

n_departments = 6
n_days = 7
n_periods = 28
SHIFT_LENGTH = 14

# ============================================================
# PENALTY SETTINGS
# ============================================================

PENALTY_SHORTAGE = 200
PENALTY_OVERHOURS = 150
PENALTY_DAYS_MIN = 300
PENALTY_SHIFT_BREAK = 100
PENALTY_NONCONSEC = 200

# ============================================================
# LOAD DEMAND FILES
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
# HELPER FUNCTIONS
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
    total_shortage = 0
    total_overwork = 0
    total_days_min = 0
    total_shift_break = 0
    total_nonconsec = 0

    for dept in range(n_departments):
        for d in range(n_days):
            for t in range(n_periods):
                assigned = np.sum(schedule[dept, d, t, :])
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

    total_fitness = (
        total_shortage +
        total_overwork +
        total_days_min +
        total_shift_break +
        total_nonconsec
    )

    return {
        "total_fitness": total_fitness,
        "shortage": total_shortage,
        "overwork": total_overwork,
        "days_min": total_days_min,
        "shift_break": total_shift_break,
        "nonconsec": total_nonconsec
    }


def compute_objectives(schedule, demand, max_hours):
    shortage = 0
    workload = 0

    for dept in range(n_departments):
        for d in range(n_days):
            for t in range(n_periods):
                shortage += max(
                    demand[dept, d, t] - np.sum(schedule[dept, d, t]),
                    0
                )

    for e in range(schedule.shape[3]):
        hours = np.sum(schedule[:, :, :, e])
        if hours > max_hours:
            workload += (hours - max_hours)

    return shortage, workload


def fitness(schedule, demand, max_hours):
    return compute_penalty_breakdown(schedule, demand, max_hours)["total_fitness"]

# ============================================================
# FFO ALGORITHM
# ============================================================

def FFO_scheduler(
    demand,
    n_employees_per_dept,
    n_fireflies,
    n_iter,
    alpha,
    beta0,
    gamma,
    max_hours,
    early_stop
):

    population = []
    fitness_vals = []
    pareto_raw = []

    best_global = float("inf")
    best_schedule = None
    no_improve = 0
    fitness_history = []

    max_emp = max(n_employees_per_dept)

    # Initialization
    for _ in range(n_fireflies):
        sched = np.random.randint(
            0, 2,
            (n_departments, n_days, n_periods, max_emp)
        )
        population.append(sched)
        fitness_vals.append(fitness(sched, demand, max_hours))

    start_time = time.time()

    for it in range(n_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness_vals[j] < fitness_vals[i]:
                    r = np.linalg.norm(population[i] - population[j])
                    beta = beta0 * np.exp(-gamma * r * r)

                    population[i] = (
                        population[i]
                        + beta * (population[j] - population[i])
                        + alpha * np.random.rand(*population[i].shape)
                    )

                    population[i] = np.clip(population[i], 0, 1)
                    population[i] = (population[i] > 0.5).astype(int)

                    fitness_vals[i] = fitness(population[i], demand, max_hours)

        for sched in population:
            pareto_raw.append(compute_objectives(sched, demand, max_hours))

        best_iter = min(fitness_vals)
        if best_iter < best_global:
            best_global = best_iter
            best_schedule = population[np.argmin(fitness_vals)].copy()
            no_improve = 0
        else:
            no_improve += 1

        fitness_history.append({
            "iteration": it + 1,
            "best": best_iter,
            "mean": np.mean(fitness_vals)
        })

        if no_improve >= early_stop:
            break

    runtime = time.time() - start_time
    pareto_filtered = pareto_filter(pareto_raw)

    return best_schedule, best_global, fitness_history, pareto_filtered, runtime

# ============================================================
# STREAMLIT SIDEBAR
# ============================================================

st.sidebar.header("FFO Parameters")

n_fireflies = st.sidebar.slider("Fireflies", 10, 50, 20)
n_iter = st.sidebar.slider("Iterations", 20, 300, 60)
alpha = st.sidebar.slider("Alpha (Randomness)", 0.1, 1.0, 0.3)
beta0 = st.sidebar.slider("Beta0 (Attractiveness)", 0.1, 2.0, 1.0)
gamma = st.sidebar.slider("Gamma (Light Absorption)", 0.01, 1.0, 0.1)
early_stop = st.sidebar.slider("Early Stop Iterations", 5, 50, 10)
max_hours = st.sidebar.slider("Max Hours / Week", 20, 60, 40)

st.sidebar.header("Employees per Department")
n_employees_per_dept = [
    st.sidebar.number_input(f"Dept {i+1} Employees", 1, 50, 20)
    for i in range(n_departments)
]

# ============================================================
# RUN BUTTON
# ============================================================

if st.sidebar.button("🔥 Run FFO"):

    best_schedule, best_score, history, pareto_data, runtime = FFO_scheduler(
        DEMAND,
        n_employees_per_dept,
        n_fireflies,
        n_iter,
        alpha,
        beta0,
        gamma,
        max_hours,
        early_stop
    )

    st.success(f"Best Fitness Score: {best_score}")
    st.info(f"Computation Time: {runtime:.2f} seconds")

    # --------------------------------------------------------
    # Convergence Plot
    # --------------------------------------------------------
    iters = [h["iteration"] for h in history]
    bests = [h["best"] for h in history]

    fig, ax = plt.subplots()
    ax.plot(iters, bests, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.set_title("FFO Fitness Convergence")
    st.pyplot(fig)

    # --------------------------------------------------------
    # Fitness Breakdown
    # --------------------------------------------------------
    st.subheader("Fitness Breakdown")
    bd = compute_penalty_breakdown(best_schedule, DEMAND, max_hours)
    st.json(bd)

    # --------------------------------------------------------
    # 🎯 Radar Chart
    # --------------------------------------------------------
    st.subheader("🎯 Constraint Balance (Radar Chart)")

    categories = [
        "Shortage",
        "Overwork",
        "Min Days",
        "Shift Break",
        "Non-Consecutive"
    ]

    values = [
        bd["shortage"],
        bd["overwork"],
        bd["days_min"],
        bd["shift_break"],
        bd["nonconsec"]
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax_radar.plot(angles, values, linewidth=2, color="#E63946")
    ax_radar.fill(angles, values, color="#E63946", alpha=0.25)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_title("Penalty Distribution (Smaller = Better)")
    st.pyplot(fig_radar)

    # --------------------------------------------------------
    # Pareto Front
    # --------------------------------------------------------
    st.subheader("Pareto Front")

    p = np.array(pareto_data)
    fig, ax = plt.subplots()
    ax.scatter(p[:, 0], p[:, 1], alpha=0.6)
    ax.set_xlabel("Total Shortage")
    ax.set_ylabel("Workload Penalty")
    st.pyplot(fig)

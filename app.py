# ============================================================
# Employee Shift Scheduling using FIXED Firefly Optimization
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt

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
# PENALTIES
# ============================================================

PENALTY_SHORTAGE = 200
PENALTY_OVERHOURS = 150
PENALTY_DAYS_MIN = 300
PENALTY_SHIFT_BREAK = 100
PENALTY_NONCONSEC = 200

# ============================================================
# LOAD DEMAND
# ============================================================

DEMAND = np.zeros((n_departments, n_days, n_periods), dtype=int)
folder_path = "./Demand/"

for dept in range(n_departments):
    file_path = os.path.join(folder_path, f"Dept{dept+1}.xlsx")
    if os.path.exists(file_path):
        df = pd.read_excel(file_path, header=None)
        df = df.iloc[1:1+n_days, 1:1+n_periods]
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        DEMAND[dept] = df.values
    else:
        st.sidebar.error(f"Dept{dept+1}.xlsx not found")

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
    shortage = overwork = days_min = shift_break = nonconsec = 0

    for dept in range(n_departments):
        for d in range(n_days):
            for t in range(n_periods):
                req = demand[dept, d, t]
                ass = np.sum(schedule[dept, d, t])
                if ass < req:
                    shortage += (req - ass) * PENALTY_SHORTAGE

        for e in range(schedule.shape[3]):
            hours = np.sum(schedule[:, :, :, e])
            if hours > max_hours:
                overwork += np.log1p(hours - max_hours) * PENALTY_OVERHOURS

            days = np.sum(np.sum(schedule[:, :, :, e], axis=2) > 0)
            if days < 6:
                days_min += PENALTY_DAYS_MIN

        for d in range(n_days):
            for e in range(schedule.shape[3]):
                daily = schedule[dept, d, :, e]
                worked = np.sum(daily)
                if worked > 0 and worked != SHIFT_LENGTH:
                    shift_break += PENALTY_SHIFT_BREAK
                if worked == SHIFT_LENGTH and longest_consecutive_ones(daily) < SHIFT_LENGTH:
                    nonconsec += PENALTY_NONCONSEC

    return {
        "total": shortage + overwork + days_min + shift_break + nonconsec,
        "shortage": shortage,
        "overwork": overwork,
        "days_min": days_min,
        "shift_break": shift_break,
        "nonconsec": nonconsec
    }


def compute_objectives(schedule, demand, max_hours):
    shortage = 0
    workload = 0
    for dept in range(n_departments):
        for d in range(n_days):
            for t in range(n_periods):
                shortage += max(demand[dept, d, t] - np.sum(schedule[dept, d, t]), 0)
    for e in range(schedule.shape[3]):
        hours = np.sum(schedule[:, :, :, e])
        if hours > max_hours:
            workload += hours - max_hours
    return shortage, workload


def fitness(schedule, demand, max_hours):
    return compute_penalty_breakdown(schedule, demand, max_hours)["total"]

# ============================================================
# FIXED FFO
# ============================================================

def FFO_scheduler(demand, n_fireflies, n_iter, alpha, beta0, gamma, max_hours, early_stop):

    max_emp = 20
    pop = np.random.rand(n_fireflies, n_departments, n_days, n_periods, max_emp)
    fitness_vals = [fitness((p > 0.5).astype(int), demand, max_hours) for p in pop]

    history = []
    pareto_raw = []
    best_global = float("inf")
    no_improve = 0

    for it in range(n_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness_vals[j] < fitness_vals[i]:
                    r = np.linalg.norm(pop[i] - pop[j])
                    beta = beta0 * np.exp(-gamma * r * r)
                    pop[i] = pop[i] + beta * (pop[j] - pop[i]) + alpha * np.random.rand(*pop[i].shape)

            # mutation
            mutation = np.random.rand(*pop[i].shape) < 0.01
            pop[i][mutation] = np.random.rand(np.sum(mutation))

            # probabilistic binarization
            bin_sched = (np.random.rand(*pop[i].shape) < pop[i]).astype(int)
            fitness_vals[i] = fitness(bin_sched, demand, max_hours)

            pareto_raw.append(compute_objectives(bin_sched, demand, max_hours))

        best_iter = min(fitness_vals)
        history.append(best_iter)

        if best_iter < best_global:
            best_global = best_iter
            best_schedule = bin_sched.copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stop:
            break

    return best_schedule, best_global, history, pareto_filter(pareto_raw)

# ============================================================
# STREAMLIT UI
# ============================================================

st.sidebar.header("FFO Parameters")
n_fireflies = st.sidebar.slider("Fireflies", 10, 40, 20)
n_iter = st.sidebar.slider("Iterations", 30, 300, 100)
alpha = st.sidebar.slider("Alpha", 0.1, 1.0, 0.3)
beta0 = st.sidebar.slider("Beta0", 0.5, 2.0, 1.0)
gamma = st.sidebar.slider("Gamma", 0.01, 1.0, 0.1)
early_stop = st.sidebar.slider("Early Stop", 10, 50, 20)
max_hours = st.sidebar.slider("Max Hours / Week", 30, 60, 40)

if st.sidebar.button("🔥 Run FFO"):
    best_sched, best_score, history, pareto = FFO_scheduler(
        DEMAND, n_fireflies, n_iter, alpha, beta0, gamma, max_hours, early_stop
    )

    st.success(f"Best Fitness: {best_score}")

    fig, ax = plt.subplots()
    ax.plot(history, marker="o")
    ax.set_title("FFO Fitness Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    st.pyplot(fig)

    bd = compute_penalty_breakdown(best_sched, DEMAND, max_hours)
    st.subheader("Fitness Breakdown")
    st.json(bd)

    # Radar (normalized)
    st.subheader("🎯 Constraint Balance (Radar Chart)")
    labels = ["Shortage", "Overwork", "Min Days", "Shift Break", "Non-Consec"]
    raw = [bd["shortage"], bd["overwork"], bd["days_min"], bd["shift_break"], bd["nonconsec"]]
    norm = [v / max(raw) for v in raw]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    norm += norm[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, norm)
    ax.fill(angles, norm, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    st.pyplot(fig)

    # Pareto
    st.subheader("Pareto Front")
    p = np.array(pareto)
    fig, ax = plt.subplots()
    ax.scatter(p[:, 0], p[:, 1])
    ax.set_xlabel("Total Shortage")
    ax.set_ylabel("Workload Penalty")
    st.pyplot(fig)

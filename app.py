# For FFO algorithm
# Employee Shift Scheduling 

import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt

#config
st.title(" Employee Shift Scheduling (FFO) ")

n_departments = 6
n_days = 7
n_periods = 28
SHIFT_LENGTH = 14

# Penalti
PENALTY_SHORTAGE = 200 # if assigned staff < demand
PENALTY_OVERHOURS = 150 # if workload staff hour > weekly max
PENALTY_DAYS_MIN = 300 # if staff work < 6 day
PENALTY_SHIFT_BREAK = 100 # if staff work < 14 period (7 hour)
PENALTY_NONCONSEC = 200 # if staff work 14 period but not in a row

# LOAD DEMAND

DEMAND = np.zeros((n_departments, n_days, n_periods), dtype=int) 
folder_path = "./Demand/" # file name is Demand

for dept in range(n_departments):
    file_path = os.path.join(folder_path, f"Dept{dept+1}.xlsx")
    if not os.path.exists(file_path):
        st.sidebar.error(f"❌ Dept{dept+1}.xlsx not found")
        continue
    df = pd.read_excel(file_path, header=None) # ignore header
    df_subset = df.iloc[1:1+n_days, 1:1+n_periods] # take 7 row of day, 28 coloumn of period
    df_subset = df_subset.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    DEMAND[dept] = df_subset.values # store demand in dept

# HELPER FUNCTIONS 

def longest_consecutive_ones(arr): #to check shift breack / consequtive
    max_len = curr = 0
    for v in arr:
        if v == 1:
            curr += 1
            max_len = max(max_len, curr)
        else:
            curr = 0
    return max_len

def pareto_filter(points): # take non-dominont solution only
    pareto = []
    for p in points:
        if not any((q[0] <= p[0] and q[1] <= p[1]) and q != p for q in points):
            pareto.append(p)
    return pareto

def compute_penalty_breakdown(schedule, demand, max_hours): # total up all constraint penalty
    total_shortage = 0
    total_overwork = 0
    total_days_min = 0
    total_shift_break = 0
    total_nonconsec = 0

    n_departments, days, periods, employees = schedule.shape

    for dept in range(n_departments):
        for d in range(days):
            for t in range(periods):
                assigned = np.sum(schedule[dept,d,t,:])
                required = demand[dept,d,t]
                if assigned < required:
                    total_shortage += (required - assigned) * PENALTY_SHORTAGE # 200

        for e in range(employees):
            total_hours = np.sum(schedule[:, :, :, e])
            if total_hours > max_hours:
                total_overwork += (total_hours - max_hours) * PENALTY_OVERHOURS # 150

            days_worked = np.sum(np.sum(schedule[:, :, :, e], axis=2) > 0)
            if days_worked < (n_days - 1):
                total_days_min += PENALTY_DAYS_MIN # 300

        for d in range(days):
            for e in range(employees):
                daily = schedule[dept,d,:,e]
                worked = np.sum(daily)
                if worked > 0 and worked != SHIFT_LENGTH: 
                    total_shift_break += PENALTY_SHIFT_BREAK #100
                if worked == SHIFT_LENGTH and longest_consecutive_ones(daily) < SHIFT_LENGTH:
                    total_nonconsec += PENALTY_NONCONSEC #200

    total_fitness = total_shortage + total_overwork + total_days_min + total_shift_break + total_nonconsec
    return {
        "total_fitness": total_fitness,
        "shortage": total_shortage,
        "overwork": total_overwork,
        "days_min": total_days_min,
        "shift_break": total_shift_break,
        "nonconsec": total_nonconsec
    }

# for pareto 
def compute_objectives(schedule, demand, max_hours):
    total_shortage = 0
    workload_penalty = 0
    n_departments, days, periods, employees = schedule.shape
    for dept in range(n_departments):
        for d in range(days):
            for t in range(periods):
                total_shortage += max(demand[dept,d,t] - np.sum(schedule[dept,d,t]), 0)
        for e in range(employees):
            total_hours = np.sum(schedule[:,:,:,e])
            if total_hours > max_hours:
                workload_penalty += (total_hours - max_hours)
    return total_shortage, workload_penalty

# for single objective
def fitness(schedule, demand, max_hours):
    return compute_penalty_breakdown(schedule,demand,max_hours)["total_fitness"]

# for all staff to at least have 1 day off (6 day work)
def generate_min_one_off_schedule(n_employees, n_days):
    off = np.zeros((n_employees, n_days), dtype=int)
    for e in range(n_employees):
        off[e, random.randint(0,n_days-1)] = 1
    return off


# FFO SCHEDULER

def FFO_scheduler(demand, n_employees_per_dept, n_Firefly, n_iter,
                  alpha, evaporation, Q, max_hours, early_stop):

    pheromone = np.ones((n_departments, n_days, 2, max(n_employees_per_dept))) # pheromone memory 
    fitness_history = [] # to plot convergence graph
    pareto_raw = [] # to store all pareto pair (total_shortage, workload_penalty)
    pareto_schedules = [] # store original schedule for each Pareto plot
    best_score_global = float("inf") # store best fitness in all iteration for early stop and convergence
    best_schedule_global = None # store best schedule with lowest fitness - not necessarily pareto-optimal
    best_off_schedules_global = None # store off schedule related
    no_improve = 0 # count for early stopping
    start_time = time.time() # to compute run time

    for it in range(n_iter):  # for each iteration
        all_scores_iter = []

        iteration_best_score = float("inf")
        iteration_best_schedule = None

        for _ in range(n_Firefly): # for each ant
            schedule = np.zeros((n_departments, n_days, n_periods, max(n_employees_per_dept)))
            off_schedules = []

            for dept in range(n_departments):
                n_emp = n_employees_per_dept[dept]
                off = generate_min_one_off_schedule(n_emp, n_days)
                off_schedules.append(off)
                for d in range(n_days):
                    for e in range(n_emp):
                        if off[e,d]==1 or random.random()<REST_PROB:
                            continue
                        tau_m = pheromone[dept,d,0,e]**alpha
                        tau_e = pheromone[dept,d,1,e]**alpha
                        p_m = tau_m/(tau_m + tau_e + 1e-6)
                        if random.random() < p_m:
                            schedule[dept,d,0:SHIFT_LENGTH,e] = 1
                        else:
                            schedule[dept,d,14:14+SHIFT_LENGTH,e] = 1

            score = fitness(schedule, demand, max_hours)
            s, w = compute_objectives(schedule, demand, max_hours)
            pareto_raw.append((s,w))
            pareto_schedules.append(schedule.copy())
            all_scores_iter.append(score)

            if score < iteration_best_score:
                iteration_best_score = score
                iteration_best_schedule = schedule.copy()

            pheromone *= (1-evaporation)
            pheromone += Q/(1+score)

        # Update global best per iteration
        if iteration_best_score < best_score_global:
            best_score_global = iteration_best_score
            best_schedule_global = iteration_best_schedule.copy()
            best_off_schedules_global = off_schedules.copy()
            no_improve = 0
        else:
            no_improve += 1

        fitness_history.append({
            "iteration": it+1,
            "best": iteration_best_score,
            "mean": np.mean(all_scores_iter),
            "worst": np.max(all_scores_iter)
        })

        if no_improve >= early_stop:
            break

    # Pareto filter
    pareto_filtered = pareto_filter(pareto_raw)
    filtered_schedules = [pareto_schedules[i] for i,p in enumerate(pareto_raw) if p in pareto_filtered]

    # Pilih best dari Pareto
    best_score_from_pareto = float("inf")
    best_schedule_final = None
    best_off_final = None
    best_index = None
    for idx, sched in enumerate(filtered_schedules):
        score = fitness(sched, demand, max_hours)
        if score < best_score_from_pareto:
            best_score_from_pareto = score
            best_schedule_final = sched.copy()
            best_off_final = best_off_schedules_global
            best_index = idx

    run_time = time.time() - start_time
    return best_schedule_final, best_score_from_pareto, fitness_history, pareto_filtered, run_time, best_off_final, best_index

# STREAMLIT CONTROLS

st.sidebar.header("FFO Parameters")
n_Firefly = st.sidebar.slider("Firefly", 5,50,20)
n_iter = st.sidebar.slider("Iterations", 10,500,50)
early_stop = st.sidebar.slider("Early Stop Iterations",1,50,10)
alpha = st.sidebar.slider("Alpha", 0.1,5.0,1.0) # show how much trust to previous experience
evaporation = st.sidebar.slider("Evaporation",0.01,0.9,0.3)
Q = st.sidebar.slider("Q",1,100,50) # show how much appreciate good solution
REST_PROB = st.sidebar.slider("Rest Probability (REST_PROB)",0.0, 0.8, 0.35, step=0.05)
max_hours = st.sidebar.slider("Max Hours / Week",20,60,40)

# for future admin to define
st.sidebar.header("Employees per Department")
n_employees_per_dept = [
    st.sidebar.number_input(f"Dept {i+1} Employees",1,50,20) for i in range(n_departments)
]

# RUN FFO

if st.sidebar.button("Run FFO"):
    best_schedule, best_score, fitness_history, pareto_data, run_time, best_off_schedules, best_idx = \
        FFO_scheduler(DEMAND, n_employees_per_dept, n_Firefly, n_iter,
                      alpha, evaporation, Q, max_hours, early_stop)

    st.session_state.best_schedule = best_schedule
    st.session_state.best_off_schedules = best_off_schedules

    st.success(f"Best Fitness Score (from Pareto): {best_score:.2f}")
    st.info(f"Computation Time: {run_time:.2f} seconds")

    # Fitness Convergence 

    iters = [int(x["iteration"]) for x in fitness_history]
    best = [x["best"] for x in fitness_history]

    fig, ax = plt.subplots()
    ax.plot(iters, best, marker='o', color='blue', label="Best Fitness per Iteration")

    # Highlight overall best fitness
    min_fitness = min(best)
    min_index = best.index(min_fitness)
    ax.plot(iters[min_index], min_fitness, marker='o', color='red', markersize=10, label="Overall Best Fitness")

    # Highlight last iteration (stop)
    ax.axvline(iters[-1], color='green', linestyle='--', label="Stop Iteration")

    ax.set_xticks(iters)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness Convergence (Best Fitness)")
    ax.legend()
    st.pyplot(fig)

    st.write(f"Algorithm stopped at iteration: {iters[-1]}")
    st.write(f"Overall Best Fitness: {min_fitness} at iteration {iters[min_index]}")


    # Pareto Front

    st.subheader("Pareto Front") # show the non-dominont solution in all iteration
    p = np.array(pareto_data)
    fig, ax = plt.subplots()
    ax.scatter(p[:,0], p[:,1], alpha=0.6, label="Pareto points")

    # choose best schedule among the non-dominont solutions
    if best_idx is not None:
        selected = p[best_idx]
        ax.scatter(selected[0], selected[1], color='red', s=100, label="Chosen Best Schedule")
    ax.set_xlabel("Total Shortage")
    ax.set_ylabel("Workload Penalty")
    ax.legend()
    st.pyplot(fig)

    # Fitness Breakdown

    st.subheader("Fitness Breakdown")
    breakdown = compute_penalty_breakdown(best_schedule, DEMAND, max_hours)
    st.json(breakdown)

    # DISPLAY SCHEDULE + HEATMAP PER DEPARTMENT

    st.subheader("Department Schedule & Heatmap")
    shift_mapping = {"08:00-15:00": range(0, SHIFT_LENGTH),
                     "15:00-22:00": range(14, 14+SHIFT_LENGTH)}

    summary_rows = []
    for dept in range(n_departments):
        n_emp = n_employees_per_dept[dept]
        employee_ids = [f"E{i+1}" for i in range(n_emp)]
        off_schedule = best_off_schedules[dept]

        st.markdown(f"### Department {dept+1}")
        rows = []
        heatmap_data = np.zeros((n_days, len(shift_mapping)))
        total_shortage_dept = 0

        for d in range(n_days):
            for idx, (shift_label, period_range) in enumerate(shift_mapping.items()):
                assigned_emps = set()
                shortage_total_shift = 0
                shortage_periods = {}

                for t in period_range:
                    if t >= n_periods: continue
                    assigned = [employee_ids[e] for e in range(n_emp) if best_schedule[dept,d,t,e]==1]
                    assigned_emps.update(assigned)
                    shortage = DEMAND[dept,d,t] - len(assigned)
                    if shortage > 0:
                        shortage_periods[f"P{t+1}"] = shortage
                        shortage_total_shift += shortage

                off_today = [employee_ids[e] for e in range(n_emp) if off_schedule[e,d]==1]
                heatmap_data[d, idx] = shortage_total_shift
                total_shortage_dept += shortage_total_shift

                rows.append([f"Day {d+1}", shift_label,
                             ", ".join(sorted(assigned_emps)) or "-",
                             ", ".join(off_today) or "-",
                             ", ".join([f"{k}({v})" for k,v in shortage_periods.items()]) or "-"])

        df_dept = pd.DataFrame(rows, columns=["Day","Shift","Employees Assigned","Employee Off","Shortage (People per Period)"])
        st.dataframe(df_dept.style.applymap(lambda v: "background-color:red;color:white" if v!="-"
                                            else "", subset=["Shortage (People per Period)"]),
                     use_container_width=True)

        st.markdown(f"**Total Shortage for Department {dept+1}: {total_shortage_dept} people**")
        summary_rows.append([f"Department {dept+1}", total_shortage_dept])

        # Heatmap
        st.markdown(f"Shortage Heatmap - Dept {dept+1}")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.imshow(heatmap_data, cmap="Reds", aspect="auto")
        ax.set_xticks(range(len(shift_mapping)))
        ax.set_xticklabels(list(shift_mapping.keys()))
        ax.set_yticks(range(n_days))
        ax.set_yticklabels([f"Day {i+1}" for i in range(n_days)])
        for i in range(n_days):
            for j in range(len(shift_mapping)):
                ax.text(j,i,int(heatmap_data[i,j]),ha="center",va="center")
        st.pyplot(fig)

    # Summary Total Shortage per Department
    st.subheader("Summary Total Shortage per Department")
    df_summary = pd.DataFrame(summary_rows, columns=["Department","Total Shortage (People)"])
    st.dataframe(df_summary,use_container_width=True)

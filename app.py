import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt

#config
st.title(" Employee Shift Scheduling (FFO) 萤火虫 ")

n_departments = 6
n_days = 7
n_periods = 28
SHIFT_LENGTH = 14

# Penalti
PENALTY_SHORTAGE = 200 
PENALTY_OVERHOURS = 150 
PENALTY_DAYS_MIN = 300 
PENALTY_SHIFT_BREAK = 100 
PENALTY_NONCONSEC = 200 

# LOAD DEMAND
DEMAND = np.zeros((n_departments, n_days, n_periods), dtype=int) 
folder_path = "./Demand/" # Ensure this folder exists with Dept1.xlsx to Dept6.xlsx

# Create dummy data if folder doesn't exist for demonstration purposes
if not os.path.exists(folder_path):
    st.warning("⚠️ Demand folder not found. Using Random Dummy Data.")
    DEMAND = np.random.randint(1, 5, size=(n_departments, n_days, n_periods))
else:
    for dept in range(n_departments):
        file_path = os.path.join(folder_path, f"Dept{dept+1}.xlsx")
        if not os.path.exists(file_path):
            st.sidebar.error(f"❌ Dept{dept+1}.xlsx not found")
            continue
        df = pd.read_excel(file_path, header=None)
        df_subset = df.iloc[1:1+n_days, 1:1+n_periods]
        df_subset = df_subset.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        DEMAND[dept] = df_subset.values

# HELPER FUNCTIONS 

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

    n_departments, days, periods, employees = schedule.shape

    for dept in range(n_departments):
        for d in range(days):
            for t in range(periods):
                assigned = np.sum(schedule[dept,d,t,:])
                required = demand[dept,d,t]
                if assigned < required:
                    total_shortage += (required - assigned) * PENALTY_SHORTAGE

        for e in range(employees):
            total_hours = np.sum(schedule[:, :, :, e])
            if total_hours > max_hours:
                total_overwork += (total_hours - max_hours) * PENALTY_OVERHOURS

            days_worked = np.sum(np.sum(schedule[:, :, :, e], axis=2) > 0)
            if days_worked < (n_days - 1):
                total_days_min += PENALTY_DAYS_MIN

        for d in range(days):
            for e in range(employees):
                daily = schedule[dept,d,:,e]
                worked = np.sum(daily)
                if worked > 0 and worked != SHIFT_LENGTH: 
                    total_shift_break += PENALTY_SHIFT_BREAK 
                if worked == SHIFT_LENGTH and longest_consecutive_ones(daily) < SHIFT_LENGTH:
                    total_nonconsec += PENALTY_NONCONSEC

    total_fitness = total_shortage + total_overwork + total_days_min + total_shift_break + total_nonconsec
    return {
        "total_fitness": total_fitness,
        "shortage": total_shortage,
        "overwork": total_overwork,
        "days_min": total_days_min,
        "shift_break": total_shift_break,
        "nonconsec": total_nonconsec
    }

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

def fitness(schedule, demand, max_hours):
    return compute_penalty_breakdown(schedule,demand,max_hours)["total_fitness"]

def generate_min_one_off_schedule(n_employees, n_days):
    off = np.zeros((n_employees, n_days), dtype=int)
    for e in range(n_employees):
        off[e, random.randint(0,n_days-1)] = 1
    return off

# FFO SPECIFIC HELPERS

def decode_position_to_schedule(position, off_schedules, n_employees_per_dept, rest_prob):
    """
    Convert continuous FFO position [0,1] to binary schedule.
    Position shape: (n_dept, n_days, max_employees)
    Thresholds:
      - If masked by off_schedule: REST
      - If val < rest_prob: REST
      - If val < rest_prob + (1-rest_prob)/2: Shift 1 (Morning)
      - Else: Shift 2 (Evening)
    """
    max_emps = max(n_employees_per_dept)
    schedule = np.zeros((n_departments, n_days, n_periods, max_emps), dtype=int)
    
    threshold_shift1 = rest_prob + (1.0 - rest_prob) / 2.0
    
    for dept in range(n_departments):
        n_emp = n_employees_per_dept[dept]
        off = off_schedules[dept]
        
        for d in range(n_days):
            for e in range(n_emp):
                val = position[dept, d, e]
                
                # Check forced day off
                if off[e, d] == 1:
                    continue # Stay 0 (Rest)
                
                # Check FFO probability
                if val < rest_prob:
                    continue # Rest
                elif val < threshold_shift1:
                    schedule[dept, d, 0:SHIFT_LENGTH, e] = 1 # Shift 1
                else:
                    schedule[dept, d, 14:14+SHIFT_LENGTH, e] = 1 # Shift 2
                    
    return schedule

# FFO SCHEDULER

def FFO_scheduler(demand, n_employees_per_dept, n_fireflies, n_iter,
                  alpha_base, beta_base, gamma, max_hours, early_stop, rest_prob):

    max_emps = max(n_employees_per_dept)
    
    # Initialize Fireflies
    # Position: Continuous values [0, 1] representing preference for shifts
    # Shape: (n_fireflies, n_departments, n_days, max_emps)
    positions = np.random.rand(n_fireflies, n_departments, n_days, max_emps)
    
    # Each firefly gets a fixed 'Off Day' pattern to maintain consistency during movement
    # List of lists of off-matrices per dept
    population_off_schedules = [] 
    for _ in range(n_fireflies):
        dept_offs = []
        for dept in range(n_departments):
            dept_offs.append(generate_min_one_off_schedule(n_employees_per_dept[dept], n_days))
        population_off_schedules.append(dept_offs)

    fitness_history = [] 
    pareto_raw = [] 
    pareto_schedules = [] 
    
    best_score_global = float("inf") 
    best_schedule_global = None 
    best_off_schedules_global = None
    no_improve = 0 
    start_time = time.time() 

    # Current fitness scores for population
    current_fitness = np.zeros(n_fireflies)
    current_schedules = [None] * n_fireflies

    # Evaluate Initial Population
    for i in range(n_fireflies):
        sched = decode_position_to_schedule(positions[i], population_off_schedules[i], n_employees_per_dept, rest_prob)
        score = fitness(sched, demand, max_hours)
        current_fitness[i] = score
        current_schedules[i] = sched
        
        # Pareto tracking
        s, w = compute_objectives(sched, demand, max_hours)
        pareto_raw.append((s,w))
        pareto_schedules.append(sched.copy())
        
        if score < best_score_global:
            best_score_global = score
            best_schedule_global = sched.copy()
            best_off_schedules_global = population_off_schedules[i]

    # Main Loop
    for it in range(n_iter):
        
        # Alpha decay (optional, helps convergence)
        alpha = alpha_base * (0.97 ** it)
        
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                # Move i towards j if j is brighter (lower fitness value)
                if current_fitness[j] < current_fitness[i]:
                    
                    # Calculate distance (Euclidean on the simplified position matrix)
                    # We flatten to calculate norm
                    pos_i_flat = positions[i].flatten()
                    pos_j_flat = positions[j].flatten()
                    dist = np.linalg.norm(pos_i_flat - pos_j_flat)
                    
                    # Calculate Attractiveness
                    beta = beta_base * np.exp(-gamma * (dist ** 2))
                    
                    # Random movement vector
                    rand_move = alpha * (np.random.rand(*positions[i].shape) - 0.5)
                    
                    # Update Position
                    positions[i] += beta * (positions[j] - positions[i]) + rand_move
                    
                    # Clamp to [0, 1]
                    positions[i] = np.clip(positions[i], 0.0, 1.0)
                    
                    # Re-evaluate i
                    new_sched = decode_position_to_schedule(positions[i], population_off_schedules[i], n_employees_per_dept, rest_prob)
                    new_score = fitness(new_sched, demand, max_hours)
                    
                    current_fitness[i] = new_score
                    current_schedules[i] = new_sched
                    
                    # Update global best
                    if new_score < best_score_global:
                        best_score_global = new_score
                        best_schedule_global = new_sched.copy()
                        best_off_schedules_global = population_off_schedules[i]
                        no_improve = 0
                        
                    # Add to pareto
                    s, w = compute_objectives(new_sched, demand, max_hours)
                    pareto_raw.append((s,w))
                    pareto_schedules.append(new_sched.copy())

        if no_improve > 0:
            no_improve += 1 # Incremented in loop only if global didn't update

        fitness_history.append({
            "iteration": it+1,
            "best": best_score_global,
            "mean": np.mean(current_fitness),
            "worst": np.max(current_fitness)
        })

        if no_improve >= early_stop:
            break

    # Pareto Filter Final
    pareto_filtered = pareto_filter(pareto_raw)
    
    # Re-map filtered points to schedules
    # Note: pareto_raw can get very large, optimization: only store unique or best few? 
    # For now keeping logic same as original for consistency
    filtered_schedules = []
    # Create a simple lookup map to avoid O(N^2)
    pareto_set = set(pareto_filtered)
    for idx, p in enumerate(pareto_raw):
        if p in pareto_set:
            filtered_schedules.append(pareto_schedules[idx])
            # Limit memory usage if too many points
            if len(filtered_schedules) > 100: break 

    # Choose best from Pareto
    best_score_from_pareto = float("inf")
    best_schedule_final = best_schedule_global # Default to global best
    best_off_final = best_off_schedules_global
    best_index = None
    
    # Sometimes global best isn't in final pareto set if dominated later, 
    # but we check the filtered list for the absolute best fitness
    for idx, sched in enumerate(filtered_schedules):
        score = fitness(sched, demand, max_hours)
        if score < best_score_from_pareto:
            best_score_from_pareto = score
            best_schedule_final = sched.copy()
            # Note: We lose the specific off-schedule mapping for pareto history items 
            # in this simple implementation, so we default to the global best's off-schedule
            # or we stick with the global best found during iteration.
            best_index = idx

    # If the global best found during loop is better than pareto selection, keep global best
    if best_score_global < best_score_from_pareto:
         best_score_from_pareto = best_score_global
         best_schedule_final = best_schedule_global
         best_off_final = best_off_schedules_global

    run_time = time.time() - start_time
    return best_schedule_final, best_score_from_pareto, fitness_history, pareto_filtered, run_time, best_off_final, best_index


# STREAMLIT CONTROLS

st.sidebar.header("FFO Parameters")
n_fireflies = st.sidebar.slider("Fireflies (Population)", 5, 50, 15)
n_iter = st.sidebar.slider("Iterations", 10, 200, 30)
early_stop = st.sidebar.slider("Early Stop", 1, 50, 10)

st.sidebar.subheader("Physics")
alpha_base = st.sidebar.slider("Alpha (Randomness)", 0.0, 1.0, 0.2)
beta_base = st.sidebar.slider("Beta0 (Attractiveness)", 0.1, 2.0, 1.0)
gamma = st.sidebar.slider("Gamma (Absorption)", 0.01, 1.0, 0.1)

st.sidebar.subheader("Constraints")
REST_PROB = st.sidebar.slider("Rest Probability Threshold", 0.0, 0.8, 0.35, step=0.05)
max_hours = st.sidebar.slider("Max Hours / Week", 20, 60, 40)

st.sidebar.header("Employees per Department")
n_employees_per_dept = [
    st.sidebar.number_input(f"Dept {i+1} Employees", 1, 50, 20) for i in range(n_departments)
]

# RUN FFO

if st.sidebar.button("Run FFO"):
    
    with st.spinner("Fireflies are optimizing... 🦗💡"):
        best_schedule, best_score, fitness_history, pareto_data, run_time, best_off_schedules, best_idx = \
            FFO_scheduler(DEMAND, n_employees_per_dept, n_fireflies, n_iter,
                          alpha_base, beta_base, gamma, max_hours, early_stop, REST_PROB)

    st.session_state.best_schedule = best_schedule
    st.session_state.best_off_schedules = best_off_schedules

    st.success(f"Best Fitness Score: {best_score:.2f}")
    st.info(f"Computation Time: {run_time:.2f} seconds")

    # Fitness Convergence 

    iters = [int(x["iteration"]) for x in fitness_history]
    best = [x["best"] for x in fitness_history]

    fig, ax = plt.subplots()
    ax.plot(iters, best, marker='o', color='gold', label="Best Fitness")
    
    min_fitness = min(best)
    min_index = best.index(min_fitness)
    ax.plot(iters[min_index], min_fitness, marker='*', color='red', markersize=15, label="Global Optima")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness (Penalty)")
    ax.set_title("FFO Convergence")
    ax.legend()
    st.pyplot(fig)

    # Pareto Front

    st.subheader("Pareto Front") 
    if len(pareto_data) > 0:
        p = np.array(pareto_data)
        fig, ax = plt.subplots()
        ax.scatter(p[:,0], p[:,1], alpha=0.6, c='orange', label="Solutions")
        
        ax.set_xlabel("Total Shortage")
        ax.set_ylabel("Workload Penalty")
        ax.set_title("Objectives Trade-off")
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("No pareto points generated.")

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
        
        # Use the specific off schedule associated with the best firefly
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

        summary_rows.append([f"Department {dept+1}", total_shortage_dept])

        # Heatmap
        st.markdown(f"**Shortage Heatmap - Dept {dept+1}**")
        fig, ax = plt.subplots(figsize=(6,3))
        im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(shift_mapping)))
        ax.set_xticklabels(list(shift_mapping.keys()))
        ax.set_yticks(range(n_days))
        ax.set_yticklabels([f"Day {i+1}" for i in range(n_days)])
        plt.colorbar(im)
        for i in range(n_days):
            for j in range(len(shift_mapping)):
                ax.text(j,i,int(heatmap_data[i,j]),ha="center",va="center")
        st.pyplot(fig)

    st.subheader("Summary Total Shortage per Department")
    df_summary = pd.DataFrame(summary_rows, columns=["Department","Total Shortage (People)"])
    st.dataframe(df_summary,use_container_width=True)

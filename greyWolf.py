import numpy as np


num_tasks = 10
num_machines = 3
task_times = np.random.randint(2, 10, size=num_tasks)  # random processing times

def calculate_makespan(solution):
    """solution = array of machine indices for each task"""
    machine_loads = np.zeros(num_machines)
    for task_idx, machine_idx in enumerate(solution):
        machine_loads[machine_idx] += task_times[task_idx]
    return np.max(machine_loads)  # makespan


num_wolves = 20
max_iter = 50

population = np.random.randint(0, num_machines, size=(num_wolves, num_tasks))
fitness = np.array([calculate_makespan(sol) for sol in population])

alpha_idx = np.argmin(fitness)
X_alpha = population[alpha_idx].copy()
alpha_score = fitness[alpha_idx]

fitness_temp = fitness.copy()
fitness_temp[alpha_idx] = np.inf
beta_idx = np.argmin(fitness_temp)
X_beta = population[beta_idx].copy()
beta_score = fitness[beta_idx]

fitness_temp[beta_idx] = np.inf
delta_idx = np.argmin(fitness_temp)
X_delta = population[delta_idx].copy()
delta_score = fitness[delta_idx]

for t in range(max_iter):
    a = 2 - t * (2 / max_iter) 

    for i in range(num_wolves):
        for d in range(num_tasks):
            r1, r2 = np.random.rand(), np.random.rand()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            D_alpha = abs(C1 * X_alpha[d] - population[i][d])
            X1 = X_alpha[d] - A1 * D_alpha

            r1, r2 = np.random.rand(), np.random.rand()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            D_beta = abs(C2 * X_beta[d] - population[i][d])
            X2 = X_beta[d] - A2 * D_beta

            r1, r2 = np.random.rand(), np.random.rand()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_delta = abs(C3 * X_delta[d] - population[i][d])
            X3 = X_delta[d] - A3 * D_delta

            # Update position
            new_pos = (X1 + X2 + X3) / 3
            # Map to machine index
            population[i][d] = int(np.clip(round(new_pos), 0, num_machines - 1))

    # Re-evaluate fitness
    fitness = np.array([calculate_makespan(sol) for sol in population])

    # Update Alpha, Beta, Delta
    alpha_idx = np.argmin(fitness)
    if fitness[alpha_idx] < alpha_score:
        X_alpha = population[alpha_idx].copy()
        alpha_score = fitness[alpha_idx]

    # Update beta and delta similarly
    fitness_temp = fitness.copy()
    fitness_temp[alpha_idx] = np.inf
    beta_idx = np.argmin(fitness_temp)
    X_beta = population[beta_idx].copy()
    beta_score = fitness[beta_idx]

    fitness_temp[beta_idx] = np.inf
    delta_idx = np.argmin(fitness_temp)
    X_delta = population[delta_idx].copy()
    delta_score = fitness[delta_idx]

print("\nTask Processing Times:", task_times)
print("Best Machine Assignment:", X_alpha)
print("Minimum Makespan Achieved:", alpha_score)

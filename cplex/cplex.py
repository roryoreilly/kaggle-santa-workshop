from docplex.mp.model import Model
from docplex.mp.progress import SolutionRecorder

import pandas as pd
import numpy as np

family_data_path = '/home/rory_o_reilly_2/santa/input/family_data.csv'
best_submission = '/home/rory_o_reilly_2/santa/input/best_submission.csv'
submissions_path = '/home/rory_o_reilly_2/santa/submissions'

data = pd.read_csv(family_data_path)
N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

family_size = data.n_people.values
days_array = np.arange(N_DAYS, 0, -1)
choice_dict = data.loc[:, 'choice_0': 'choice_9'].T.to_dict()

choice_array_num = np.full((data.shape[0], N_DAYS + 1), -1)
for i, choice in enumerate(data.loc[:, 'choice_0': 'choice_9'].values):
    for d, day in enumerate(choice):
        choice_array_num[i, day] = d
penalties_array = np.array([
    [
        0,
        50,
        50 + 9 * n,
        100 + 9 * n,
        200 + 9 * n,
        200 + 18 * n,
        300 + 18 * n,
        300 + 36 * n,
        400 + 36 * n,
        500 + 36 * n + 199 * n,
        500 + 36 * n + 398 * n
    ]
    for n in range(family_size.max() + 1)
])

def cost_function(prediction):
    penalty = 0

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = np.zeros((len(days_array)+1))
    N = family_size.shape[0]
    
    # Looping over each family; d is the day, n is size of that family, 
    # and choice is their top choices
    for i in range(N):
        # add the family member count to the daily occupancy
        n = family_size[i]
        d = prediction[i]
        choice = choice_array_num[i]
        
        daily_occupancy[d] += n

        # Calculate the penalty for not getting top preference
        penalty += penalties_array[n, choice[d]]

    choice_cost = penalty
        
    # for each date, check total occupancy
    #  (using soft constraints instead of hard constraints)
    relevant_occupancy = daily_occupancy[1:]
    incorrect_occupancy = np.any(
        (relevant_occupancy > MAX_OCCUPANCY) | 
        (relevant_occupancy < MIN_OCCUPANCY)
    )
    
    if incorrect_occupancy:
        penalty += 100000000

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    init_occupancy = daily_occupancy[days_array[0]]
    accounting_cost = (init_occupancy - 125.0) / 400.0 * init_occupancy**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)
    
    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = init_occupancy
    for day in days_array[1:]:
        today_count = daily_occupancy[day]
        diff = np.abs(today_count - yesterday_count)
        accounting_cost += max(0, (today_count - 125.0) / 400.0 * today_count**(0.5 + diff / 50.0))
        yesterday_count = today_count

    penalty += accounting_cost

    return penalty#, choice_cost, accounting_cost

class MyProgressListener(SolutionRecorder):
    def __init__(self, model, assign_family_to_day_vars):
        SolutionRecorder.__init__(self)
        self.solutions = []
        self.current_objective = 999999;
        self.assign_family_to_day_vars = assign_family_to_day_vars

    def notify_solution(self, s):
        SolutionRecorder.notify_solution(self, s)
        self.solutions.append(s)
        if self.current_progress_data.current_objective >= self.current_objective:
            return;
        self.current_objective = self.current_progress_data.current_objective;

        print ('Intermediate Solution')
        assigned_days = np.zeros(5000, int)
        for fid, day in self.assign_family_to_day_vars:
            if sol.get_value(self.assign_family_to_day_vars[fid, day]) > 0:
                assigned_days[fid] = day + 1

        solution = pd.DataFrame(assigned_days, columns =['assigned_day']).astype('int32')
        score = cost_function(assigned_days)
        print('Score: ' + str(score))
        solution.to_csv('./submissions/submission_' + str(score) + '.csv', line_terminator='\n', encoding='utf-8')
    
    def get_solutions(self):
        return self.solutions

def build_model(desired, sub, n_people): # can't run on kaggle notebooks 
    def accounting_penalty(day, next_day):
        return (day-125)*(day**(0.5 + abs(day-next_day)/50.0))/400
    
    print('Building model')

    FAMILY_COST = np.asarray([0,50,50,100,200,200,300,300,400,500])
    MEMBER_COST = np.asarray([0, 0, 9,  9,  9, 18, 18, 36, 36,235])

    num_days = desired.max()
    num_families = desired.shape[0]
    solver = Model(name='Santa2019')

    print('Adding costs')
    C = {}
    for fid, choices in enumerate(desired):
        for cid in range(5):
            C[fid, choices[cid]-1] = FAMILY_COST[cid] + n_people[fid] * MEMBER_COST[cid]

    B = solver.binary_var_dict(C, name='B')
    I = solver.integer_var_list(num_days, lb=125, ub=300, name='I')

    print('Adding people per day')
    for day in range(num_days):
        solver.add(solver.sum(n_people[fid]*B[fid, day] for fid in range(num_families) if (fid,day) in B) == I[day])

    print('Constraint of one choice per family')
    for fid in range(num_families):
        solver.add(solver.sum(B[fid, day] for day in range(num_days) if (fid,day) in B) == 1)

    print('Preference cost cal')
    preference_cost = solver.sum(C[fid, day]*B[fid, day] for fid, day in B)

    print('Accounting cost cal')
    Y = solver.binary_var_cube(num_days, 176, 176, name='Y')

    for day in range(num_days):
        next_day = np.clip(day+1, 0, num_days-1)
        gen = [(u,v) for v in range(176) for u in range(176)]
        solver.add(solver.sum(Y[day,u,v]*u for u,v in gen) == I[day]-125)
        solver.add(solver.sum(Y[day,u,v]*v for u,v in gen) == I[next_day]-125)
        solver.add(solver.sum(Y[day,u,v]   for u,v in gen) == 1)
        
    gen = [(day,u,v) for day in range(num_days) for v in range(176) for u in range(176)]
    accounting_penalties = solver.sum(accounting_penalty(u+125,v+125) * Y[day,u,v] for day,u,v in gen)
    solver.add(accounting_penalties >= 5520)
    solver.minimize(accounting_penalties+preference_cost)

    print('Setting best submission')

    sub_cst = solver.add_constraints(B[f, a-1] == 1 for f,a in zip(sub.index, sub.assigned_day))

    solver.parameters.threads = 1
    solver.parameters.mip.tolerances.mipgap = 0.00
    
    solver.print_information()

    print('Doing first solve')
    if solver.solve(log_output=True):
        print('Preparing solve has been completed')
        print('Objective sample: {}'.format(solver.objective_value))
        solver.report()

        solver.remove(sub_cst)

        listener = MyProgressListener(solver)
        solver.add_progress_listener(listener)

        # priority_sub = sub.loc[sub.index.isin(priority_fids)]
        # solver.add_constraints(B[f, a-1] == 1 for f,a in zip(priority_sub.index, priority_sub.assigned_day))

        solver.parameters.threads = 8
        solver.parameters.timelimit = 60 * 10
        solver.print_information()
        
        print('Solving...')
        sol = solver.solve(log_output=True)
        if sol:
            print(sol.objective_value)
            assigned_days = np.zeros(num_families, int)
            for fid, day in C:
                if sol[B[fid, day]] > 0:
                    assigned_days[fid] = day + 1
            return assigned_days


ds = pd.read_csv(family_data_path)
best = pd.read_csv(best_submission, index_col='family_id')
sub = build_model(ds.values[:,1:11], best, ds.values[:,11])
sub.to_csv('submission.csv')
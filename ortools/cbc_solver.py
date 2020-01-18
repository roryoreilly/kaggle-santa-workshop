from ortools.linear_solver import pywraplp
import pandas as pd

family_data_path = '/home/rory_o_reilly_2/santa/input/family_data.csv'
starting_model = '/home/rory_o_reilly_2/santa/input/santa-best-score.csv'

def example_ortools(desired, n_people, has_accounting=True):
    def accounting_penalty(day, next_day):
        return (day-125)*(day**(0.5 + abs(day-next_day)/50.0))/400
    NUM_THREADS = 8
    NUM_SECONDS = 10
    FAMILY_COST = np.asarray([0,50,50,100,200,200,300,300,400,500])
    MEMBER_COST = np.asarray([0, 0, 9,  9,  9, 18, 18, 36, 36,235])
    num_days = desired.max()
    num_families = desired.shape[0]
    solver = pywraplp.Solver('Santa2019', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    solver.set_time_limit(NUM_SECONDS*NUM_THREADS*1000)
    solver.SetNumThreads(NUM_THREADS)
    C, B, I = {}, {}, {}
    for fid, choices in enumerate(desired):
        for cid in range(10):
            B[fid, choices[cid]-1] = solver.BoolVar('')
            C[fid, choices[cid]-1] = FAMILY_COST[cid] + n_people[fid] * MEMBER_COST[cid]

    for day in range(num_days):
        I[day] = solver.IntVar(125, 300, f'I{day}')
        solver.Add(solver.Sum(n_people[fid]*B[fid, day] for fid in range(num_families) if (fid,day) in B) == I[day])

    for fid in range(num_families):
        solver.Add(solver.Sum(B[fid, day] for day in range(num_days) if (fid,day) in B) == 1)

    objective = solver.Sum(C[fid, day]*B[fid, day] for fid, day in B)
    if has_accounting:
        Y = {}

        for day in range(num_days):
            next_day = np.clip(day+1, 0, num_days-1)
            gen = [(u,v) for v in range(176) for u in range(176)]
            for u,v in gen:
                Y[day,u,v] = solver.BoolVar('')
            solver.Add(solver.Sum(Y[day,u,v]*u for u,v in gen) == I[day]-125)
            solver.Add(solver.Sum(Y[day,u,v]*v for u,v in gen) == I[next_day]-125)
            solver.Add(solver.Sum(Y[day,u,v]   for u,v in gen) == 1)
            
        accounting_penalties = solver.Sum(accounting_penalty(u+125,v+125) * Y[day,u,v] for day,u,v in Y)
        objective += accounting_penalties
        solver.Add(accounting_penalties >= 6000)
    
    for fam in range(num_families):
        submission = pd.read_csv(sample_submission_path, index_col='family_id').values
        day = submission[fam][0] -1
        model.AddDecisionStrategy([assign[(fam, day)]], cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE)

    solver.Minimize(objective)
    sol = solver.Solve()
    status = ['OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNBOUNDED', 'ABNORMAL', 'MODEL_INVALID', 'NOT_SOLVED']
    if status[sol] == 'OPTIMAL':
        print("Result: ", objective.solution_value())
        assigned_days = np.zeros(num_families, int)
        for fid, day in B:
            if B[fid, day].solution_value() > 0.5:
                assigned_days[fid] = day + 1
        return assigned_days


ds = pd.read_csv(family_data_path)
sub = example_ortools(ds.values[:,1:11], ds.values[:,11], True)
sub.to_csv('submission.csv')
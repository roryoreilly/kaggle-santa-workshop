from docloud.job import JobClient
from docplex.mp.context import Context

import sched, time
import json
import numpy as np
import pandas as pd

data = pd.read_csv('./input/family_data.csv')
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

def cost_function_detailed(prediction, penalties_array, family_size, days):
    penalty = 0

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = np.zeros((len(days)+1))
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
    init_occupancy = daily_occupancy[days[0]]
    accounting_cost = (init_occupancy - 125.0) / 400.0 * init_occupancy**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)
    
    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = init_occupancy
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = np.abs(today_count - yesterday_count)
        accounting_cost += max(0, (today_count - 125.0) / 400.0 * today_count**(0.5 + diff / 50.0))
        yesterday_count = today_count

    penalty += accounting_cost

    return penalty, choice_cost, accounting_cost

def calculate_costs():
	solution_df = pd.read_csv('./input/best_submission.csv')
	pred = solution_df['assigned_day'].values
	total_cost, choice_cost, acc_cost = cost_function_detailed(pred, penalties_array, family_size, days_array)
	print('Choice cost ({}) + Acc cost({}) = Total cost ({})'.format(choice_cost, acc_cost, total_cost))


def create_solution_from_logs():
	begun = False
	fids = np.arange(5000).reshape((5000, -1))
	av = np.zeros(5000).reshape((5000, -1))
	assigned = np.concatenate([fids, av], axis=1)
	with open("./logs.txt", "r") as f:
		for line in f:
			if '***** START' in line:
				begun = True
			elif '***** STOP' in line:
				break
			elif begun:
				entry = line.strip().split(',')
				fid = int(entry[0])
				day = int(entry[1])
				av[fid] = day
	assigned = np.concatenate([fids, av], axis=1)
	solution_df = pd.DataFrame(assigned, columns =['family_id', 'assigned_day']).astype('int32')
	solution_df.to_csv('./input/best_submission.csv', index=False)

def run_model():
	url = 'https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/'
	key = 'api_fc2d8028-3ebc-4a3a-8664-b4018b1c05a8'

	client = JobClient(url=url, api_key=key)

	resp = client.execute(input=['workshop_model.py',
			'../input/best_submission.csv',
			'../input/family_data.csv'],
		output='solution.json',
		load_solution=True,
		log='logs.txt')

if __name__ == '__main__':
	# s = sched.scheduler(time.time, time.sleep)
	# x = 0
	# for _ in range(10):
	# 	s.enter(x, 1, run_model)
	# 	s.enter(x+63, 1, create_solution_from_logs)
	# 	x += 65
	# s.run()
	create_solution_from_logs()
	calculate_costs()
	run_model()
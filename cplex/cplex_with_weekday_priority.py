from docplex.mp.model import Model
from docplex.mp.progress import SolutionRecorder

import pandas as pd
import numpy as np

family_data_path = '/home/rory_o_reilly_2/santa/input/family_data.csv'
best_submission = '/home/rory_o_reilly_2/santa/input/best_submission.csv'
submissions_path = '/home/rory_o_reilly_2/santa/submissions'

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

priority_fids = [   2,   15,   25,   26,   37,   38,   44,   46,   51,   53,   55,
         84,   86,   89,   92,  106,  108,  114,  115,  118,  120,  124,
        130,  132,  144,  146,  147,  148,  158,  163,  172,  181,  186,
        187,  190,  192,  195,  206,  222,  227,  229,  230,  231,  236,
        239,  246,  256,  263,  269,  281,  284,  288,  297,  303,  304,
        305,  312,  318,  320,  323,  333,  334,  341,  351,  352,  359,
        364,  381,  382,  385,  393,  395,  398,  404,  408,  425,  442,
        443,  446,  457,  460,  466,  473,  474,  482,  483,  485,  489,
        490,  495,  500,  501,  502,  505,  508,  509,  513,  515,  527,
        534,  536,  537,  538,  539,  542,  552,  561,  565,  571,  580,
        582,  588,  591,  593,  599,  600,  604,  611,  615,  616,  619,
        622,  629,  639,  646,  652,  657,  658,  666,  674,  679,  680,
        692,  693,  697,  699,  700,  705,  711,  716,  723,  725,  734,
        739,  742,  744,  749,  754,  758,  760,  761,  766,  780,  781,
        793,  800,  804,  806,  810,  814,  821,  822,  823,  826,  829,
        830,  831,  832,  834,  842,  849,  851,  854,  855,  860,  861,
        874,  879,  881,  885,  887,  889,  890,  893,  894,  896,  901,
        907,  924,  925,  931,  937,  942,  947,  955,  957,  959,  960,
        962,  970,  972,  974,  990,  995,  996,  997, 1003, 1018, 1025,
       1028, 1029, 1040, 1042, 1043, 1047, 1056, 1060, 1061, 1070, 1077,
       1086, 1087, 1090, 1091, 1097, 1098, 1111, 1119, 1122, 1126, 1128,
       1139, 1150, 1157, 1165, 1166, 1170, 1179, 1181, 1198, 1199, 1200,
       1208, 1211, 1217, 1223, 1229, 1237, 1239, 1243, 1251, 1253, 1266,
       1273, 1276, 1277, 1279, 1280, 1283, 1291, 1292, 1301, 1302, 1309,
       1325, 1327, 1334, 1337, 1338, 1339, 1340, 1343, 1346, 1348, 1354,
       1356, 1360, 1364, 1366, 1370, 1371, 1388, 1400, 1404, 1407, 1419,
       1421, 1422, 1426, 1430, 1431, 1433, 1438, 1439, 1440, 1446, 1449,
       1451, 1452, 1455, 1458, 1466, 1468, 1471, 1478, 1480, 1490, 1491,
       1493, 1502, 1512, 1517, 1518, 1520, 1521, 1538, 1541, 1543, 1544,
       1559, 1563, 1566, 1567, 1571, 1586, 1588, 1592, 1598, 1600, 1601,
       1603, 1604, 1606, 1609, 1618, 1621, 1623, 1625, 1634, 1639, 1640,
       1644, 1645, 1650, 1652, 1665, 1668, 1669, 1672, 1676, 1677, 1680,
       1683, 1702, 1703, 1708, 1709, 1710, 1711, 1719, 1721, 1728, 1731,
       1732, 1734, 1736, 1738, 1741, 1742, 1765, 1767, 1783, 1786, 1791,
       1794, 1795, 1800, 1801, 1803, 1804, 1806, 1812, 1824, 1834, 1839,
       1840, 1845, 1849, 1854, 1855, 1859, 1861, 1867, 1869, 1881, 1883,
       1884, 1886, 1890, 1891, 1899, 1900, 1908, 1909, 1916, 1921, 1923,
       1926, 1928, 1932, 1935, 1936, 1943, 1945, 1947, 1959, 1962, 1967,
       1969, 1978, 1979, 1983, 1990, 1991, 1999, 2009, 2010, 2013, 2015,
       2023, 2027, 2028, 2030, 2032, 2034, 2038, 2039, 2043, 2044, 2045,
       2053, 2060, 2062, 2074, 2076, 2077, 2078, 2081, 2084, 2090, 2096,
       2100, 2103, 2105, 2106, 2108, 2109, 2113, 2116, 2117, 2119, 2121,
       2130, 2133, 2134, 2136, 2143, 2144, 2150, 2152, 2154, 2156, 2157,
       2162, 2167, 2179, 2183, 2184, 2187, 2193, 2198, 2204, 2206, 2216,
       2217, 2218, 2220, 2221, 2225, 2230, 2231, 2233, 2237, 2240, 2253,
       2254, 2255, 2258, 2265, 2267, 2276, 2282, 2284, 2291, 2299, 2300,
       2304, 2310, 2312, 2315, 2320, 2329, 2331, 2332, 2333, 2336, 2337,
       2340, 2341, 2345, 2356, 2357, 2370, 2372, 2378, 2384, 2389, 2406,
       2409, 2417, 2422, 2426, 2427, 2431, 2433, 2435, 2436, 2438, 2439,
       2446, 2448, 2451, 2455, 2457, 2473, 2475, 2477, 2479, 2485, 2498,
       2502, 2503, 2504, 2510, 2512, 2522, 2524, 2533, 2534, 2538, 2556,
       2561, 2569, 2572, 2578, 2584, 2585, 2587, 2602, 2609, 2611, 2614,
       2616, 2623, 2626, 2630, 2632, 2637, 2640, 2643, 2645, 2646, 2662,
       2664, 2668, 2672, 2677, 2678, 2684, 2686, 2689, 2693, 2712, 2714,
       2717, 2718, 2735, 2736, 2739, 2741, 2750, 2752, 2753, 2765, 2768,
       2770, 2772, 2773, 2776, 2790, 2795, 2796, 2807, 2809, 2810, 2814,
       2815, 2820, 2825, 2840, 2844, 2847, 2855, 2856, 2875, 2881, 2893,
       2904, 2908, 2922, 2928, 2934, 2939, 2950, 2951, 2954, 2957, 2960,
       2962, 2963, 2969, 2971, 2977, 2988, 2989, 2992, 2994, 2996, 2998,
       2999, 3003, 3004, 3007, 3008, 3010, 3023, 3029, 3052, 3056, 3058,
       3068, 3070, 3075, 3086, 3090, 3096, 3098, 3100, 3107, 3108, 3112,
       3120, 3125, 3126, 3132, 3140, 3142, 3147, 3163, 3166, 3176, 3177,
       3183, 3196, 3201, 3204, 3209, 3211, 3213, 3216, 3219, 3221, 3258,
       3282, 3284, 3287, 3291, 3293, 3296, 3297, 3300, 3301, 3305, 3311,
       3313, 3316, 3318, 3319, 3324, 3325, 3327, 3334, 3335, 3355, 3357,
       3361, 3363, 3366, 3368, 3369, 3372, 3374, 3377, 3379, 3382, 3385,
       3398, 3399, 3402, 3405, 3407, 3409, 3411, 3416, 3420, 3425, 3428,
       3431, 3434, 3436, 3446, 3448, 3463, 3464, 3466, 3467, 3474, 3486,
       3487, 3489, 3493, 3500, 3501, 3517, 3521, 3524, 3525, 3526, 3535,
       3538, 3539, 3544, 3545, 3546, 3548, 3550, 3554, 3558, 3563, 3567,
       3570, 3571, 3573, 3582, 3585, 3592, 3593, 3602, 3606, 3616, 3617,
       3624, 3626, 3628, 3641, 3652, 3657, 3667, 3674, 3675, 3678, 3682,
       3683, 3684, 3686, 3687, 3701, 3708, 3709, 3727, 3731, 3733, 3736,
       3742, 3750, 3752, 3755, 3761, 3765, 3774, 3775, 3777, 3792, 3793,
       3797, 3801, 3802, 3805, 3807, 3811, 3816, 3819, 3827, 3830, 3836,
       3838, 3840, 3842, 3844, 3845, 3852, 3862, 3867, 3869, 3870, 3876,
       3878, 3880, 3884, 3885, 3886, 3887, 3888, 3889, 3896, 3900, 3903,
       3910, 3919, 3922, 3928, 3932, 3934, 3937, 3938, 3939, 3946, 3954,
       3959, 3960, 3963, 3965, 3968, 3973, 3975, 3976, 3981, 3982, 3985,
       3986, 3988, 3995, 3996, 3997, 4003, 4012, 4016, 4018, 4020, 4021,
       4022, 4033, 4042, 4043, 4044, 4051, 4054, 4058, 4070, 4081, 4085,
       4087, 4102, 4109, 4114, 4117, 4118, 4119, 4122, 4123, 4128, 4130,
       4131, 4132, 4138, 4143, 4144, 4148, 4152, 4157, 4158, 4164, 4166,
       4169, 4170, 4171, 4177, 4185, 4186, 4187, 4189, 4192, 4194, 4200,
       4209, 4213, 4220, 4222, 4230, 4234, 4239, 4247, 4252, 4253, 4255,
       4256, 4258, 4260, 4261, 4265, 4266, 4270, 4271, 4275, 4280, 4283,
       4293, 4299, 4304, 4307, 4311, 4319, 4323, 4326, 4332, 4333, 4337,
       4341, 4342, 4353, 4363, 4364, 4365, 4378, 4384, 4385, 4390, 4394,
       4396, 4402, 4412, 4416, 4417, 4418, 4422, 4425, 4427, 4428, 4429,
       4433, 4453, 4459, 4469, 4476, 4484, 4488, 4489, 4490, 4491, 4492,
       4493, 4494, 4497, 4498, 4502, 4504, 4511, 4517, 4524, 4532, 4534,
       4535, 4536, 4539, 4548, 4552, 4554, 4573, 4577, 4578, 4579, 4580,
       4586, 4592, 4593, 4600, 4601, 4606, 4609, 4610, 4612, 4615, 4622,
       4629, 4630, 4642, 4643, 4647, 4649, 4651, 4653, 4668, 4671, 4675,
       4676, 4677, 4684, 4688, 4689, 4692, 4699, 4705, 4708, 4711, 4717,
       4719, 4727, 4728, 4730, 4734, 4737, 4746, 4750, 4752, 4756, 4759,
       4762, 4764, 4767, 4771, 4782, 4785, 4790, 4797, 4801, 4803, 4804,
       4805, 4810, 4818, 4827, 4829, 4837, 4840, 4841, 4843, 4854, 4856,
       4858, 4862, 4867, 4871, 4876, 4878, 4882, 4885, 4887, 4891, 4899,
       4903, 4905, 4907, 4908, 4910, 4915, 4920, 4921, 4924, 4925, 4928,
       4932, 4934, 4937, 4938, 4940, 4944, 4945, 4957, 4962, 4965, 4967,
       4970, 4974, 4979, 4981, 4984, 4987, 4993, 4995, 4999]

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
            if s.get_value(self.assign_family_to_day_vars[fid, day]) > 0:
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

    print('Contraint for day choices')
    solver.add(solver.sum(B[fid, desired[fid, 1]-1] for fid in range(num_families)) <= 1000)
    solver.add(solver.sum(B[fid, desired[fid, 2]-1] for fid in range(num_families)) <= 300)
    solver.add(solver.sum(B[fid, desired[fid, 3]-1] for fid in range(num_families)) <= 100)
    solver.add(solver.sum(B[fid, desired[fid, 4]-1] for fid in range(num_families)) <= 10)

    print('Constraint of one choice per family')
    for fid in range(num_families):
        solver.add(solver.sum(B[fid, day] for day in range(num_days) if (fid,day) in B) == 1)

    print('Constraint for days with little people')
    days_low = [58, 64, 65, 70, 71, 72, 77, 78, 79, 84, 85, 86, 91, 92, 93, 98, 99, 100]
    for day in days_low:
      solver.add(I[day-1] <= 130)

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
	
        listener = MyProgressListener(solver, B)
        solver.add_progress_listener(listener)
        priority_sub = sub.loc[sub.index.isin(priority_fids)]
        solver.add_constraints(B[f, a-1] == 1 for f,a in zip(priority_sub.index, priority_sub.assigned_day))

        solver.parameters.threads = 4
        # solver.parameters.timelimit = 60 * 55
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

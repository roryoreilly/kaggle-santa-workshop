from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from docplex.util.environment import get_environment

from functools import partial, wraps
import os
from os.path import splitext
import threading

import pandas as pd
import numpy as np
import time 

FAMILY_COST = np.asarray([0,50,50,100,200,200,300,300,400,500])
MEMBER_COST = np.asarray([0, 0, 9,  9,  9, 18, 18, 36, 36,235])
NUM_DAYS = 100
NUM_FAMILIES = 5000


priority_fids = [   2,   15,   21,   25,   26,   37,   38,   44,   46,   51,   53,
         55,   77,   84,   86,   88,   89,   92,  106,  108,  114,  115,
        116,  118,  120,  124,  128,  130,  132,  144,  146,  147,  148,
        155,  158,  163,  166,  172,  181,  186,  187,  190,  192,  195,
        199,  202,  206,  210,  216,  222,  227,  229,  230,  231,  236,
        239,  246,  256,  258,  263,  269,  272,  274,  275,  281,  284,
        288,  297,  303,  304,  305,  312,  318,  320,  323,  333,  334,
        341,  351,  352,  359,  360,  364,  375,  381,  382,  385,  393,
        395,  398,  404,  408,  425,  442,  443,  446,  457,  460,  461,
        466,  469,  473,  474,  479,  482,  483,  485,  489,  490,  493,
        495,  497,  500,  501,  502,  503,  505,  508,  509,  511,  513,
        515,  527,  534,  536,  537,  538,  539,  542,  552,  561,  565,
        571,  572,  580,  581,  582,  588,  591,  593,  599,  600,  604,
        611,  615,  616,  619,  622,  629,  635,  639,  640,  646,  647,
        649,  652,  657,  658,  659,  663,  666,  672,  674,  679,  680,
        682,  691,  692,  693,  695,  697,  699,  700,  702,  705,  711,
        716,  723,  725,  734,  739,  742,  744,  749,  753,  754,  758,
        759,  760,  761,  766,  775,  780,  781,  787,  793,  800,  804,
        806,  810,  814,  821,  822,  823,  826,  829,  830,  831,  832,
        834,  842,  849,  851,  853,  854,  855,  860,  861,  874,  879,
        881,  885,  887,  889,  890,  893,  894,  896,  901,  907,  924,
        925,  931,  937,  942,  947,  950,  955,  957,  959,  960,  962,
        970,  972,  974,  985,  990,  995,  996,  997, 1003, 1018, 1025,
       1028, 1029, 1040, 1042, 1043, 1047, 1056, 1060, 1061, 1070, 1077,
       1086, 1087, 1090, 1091, 1097, 1098, 1111, 1119, 1122, 1126, 1128,
       1139, 1150, 1157, 1165, 1166, 1170, 1179, 1181, 1192, 1195, 1198,
       1199, 1200, 1205, 1208, 1211, 1213, 1217, 1223, 1224, 1229, 1237,
       1239, 1243, 1251, 1253, 1263, 1266, 1267, 1273, 1276, 1277, 1279,
       1280, 1281, 1283, 1286, 1291, 1292, 1301, 1302, 1309, 1319, 1325,
       1327, 1330, 1334, 1336, 1337, 1338, 1339, 1340, 1343, 1346, 1348,
       1354, 1356, 1360, 1362, 1364, 1366, 1370, 1371, 1388, 1390, 1396,
       1400, 1404, 1407, 1418, 1419, 1421, 1422, 1426, 1430, 1431, 1433,
       1434, 1438, 1439, 1440, 1446, 1449, 1451, 1452, 1455, 1458, 1461,
       1466, 1468, 1471, 1478, 1480, 1483, 1490, 1491, 1493, 1496, 1497,
       1502, 1511, 1512, 1517, 1518, 1520, 1521, 1538, 1540, 1541, 1542,
       1543, 1544, 1559, 1562, 1563, 1566, 1567, 1571, 1586, 1588, 1592,
       1598, 1600, 1601, 1603, 1604, 1606, 1609, 1618, 1621, 1623, 1624,
       1625, 1632, 1634, 1636, 1639, 1640, 1644, 1645, 1650, 1652, 1665,
       1668, 1669, 1672, 1674, 1676, 1677, 1680, 1683, 1687, 1702, 1703,
       1708, 1709, 1710, 1711, 1719, 1721, 1728, 1731, 1732, 1734, 1736,
       1738, 1741, 1742, 1747, 1765, 1767, 1779, 1783, 1786, 1787, 1791,
       1794, 1795, 1797, 1800, 1801, 1802, 1803, 1804, 1806, 1812, 1824,
       1834, 1839, 1840, 1845, 1849, 1854, 1855, 1859, 1861, 1865, 1867,
       1869, 1881, 1883, 1884, 1886, 1890, 1891, 1899, 1900, 1901, 1908,
       1909, 1910, 1916, 1921, 1923, 1926, 1928, 1932, 1933, 1935, 1936,
       1937, 1942, 1943, 1945, 1947, 1952, 1959, 1962, 1967, 1969, 1978,
       1979, 1980, 1983, 1987, 1990, 1991, 1999, 2009, 2010, 2013, 2015,
       2023, 2027, 2028, 2030, 2032, 2034, 2038, 2039, 2042, 2043, 2044,
       2045, 2053, 2060, 2062, 2074, 2076, 2077, 2078, 2081, 2084, 2090,
       2096, 2100, 2103, 2105, 2106, 2108, 2109, 2113, 2116, 2117, 2119,
       2121, 2130, 2133, 2134, 2136, 2143, 2144, 2150, 2152, 2154, 2156,
       2157, 2162, 2167, 2172, 2175, 2179, 2183, 2184, 2187, 2193, 2198,
       2202, 2203, 2204, 2206, 2211, 2216, 2217, 2218, 2220, 2221, 2225,
       2230, 2231, 2233, 2237, 2238, 2240, 2245, 2253, 2254, 2255, 2256,
       2258, 2265, 2267, 2274, 2276, 2282, 2284, 2291, 2296, 2299, 2300,
       2304, 2307, 2310, 2312, 2315, 2320, 2321, 2329, 2331, 2332, 2333,
       2336, 2337, 2340, 2341, 2345, 2356, 2357, 2359, 2370, 2371, 2372,
       2378, 2384, 2389, 2406, 2409, 2412, 2417, 2422, 2426, 2427, 2431,
       2433, 2434, 2435, 2436, 2438, 2439, 2441, 2443, 2446, 2448, 2451,
       2455, 2457, 2463, 2473, 2475, 2477, 2479, 2485, 2498, 2502, 2503,
       2504, 2510, 2512, 2522, 2524, 2525, 2529, 2533, 2534, 2538, 2556,
       2558, 2561, 2569, 2572, 2578, 2584, 2585, 2587, 2591, 2602, 2609,
       2611, 2614, 2616, 2623, 2626, 2630, 2632, 2637, 2640, 2641, 2643,
       2645, 2646, 2661, 2662, 2664, 2667, 2668, 2672, 2676, 2677, 2678,
       2684, 2686, 2689, 2693, 2696, 2712, 2714, 2717, 2718, 2720, 2735,
       2736, 2739, 2741, 2747, 2750, 2751, 2752, 2753, 2764, 2765, 2768,
       2770, 2772, 2773, 2776, 2777, 2790, 2795, 2796, 2807, 2809, 2810,
       2814, 2815, 2820, 2825, 2834, 2840, 2844, 2847, 2855, 2856, 2875,
       2881, 2888, 2893, 2904, 2908, 2922, 2928, 2934, 2937, 2939, 2950,
       2951, 2954, 2957, 2960, 2962, 2963, 2965, 2969, 2971, 2977, 2984,
       2988, 2989, 2992, 2994, 2996, 2998, 2999, 3001, 3002, 3003, 3004,
       3007, 3008, 3010, 3013, 3023, 3029, 3049, 3052, 3056, 3058, 3068,
       3070, 3075, 3077, 3086, 3090, 3094, 3096, 3098, 3100, 3106, 3107,
       3108, 3112, 3114, 3115, 3120, 3123, 3125, 3126, 3132, 3140, 3142,
       3145, 3147, 3154, 3163, 3166, 3176, 3177, 3183, 3196, 3201, 3204,
       3209, 3211, 3213, 3216, 3219, 3221, 3226, 3258, 3282, 3284, 3287,
       3291, 3293, 3296, 3297, 3299, 3300, 3301, 3305, 3308, 3311, 3313,
       3316, 3318, 3319, 3324, 3325, 3327, 3330, 3334, 3335, 3347, 3350,
       3355, 3357, 3361, 3363, 3366, 3368, 3369, 3372, 3374, 3377, 3379,
       3382, 3385, 3388, 3398, 3399, 3402, 3405, 3407, 3409, 3411, 3416,
       3420, 3422, 3425, 3428, 3431, 3432, 3433, 3434, 3436, 3443, 3446,
       3448, 3463, 3464, 3466, 3467, 3474, 3482, 3486, 3487, 3489, 3493,
       3496, 3500, 3501, 3517, 3521, 3524, 3525, 3526, 3535, 3538, 3539,
       3540, 3544, 3545, 3546, 3548, 3550, 3554, 3558, 3563, 3565, 3567,
       3570, 3571, 3573, 3579, 3582, 3585, 3592, 3593, 3602, 3606, 3608,
       3616, 3617, 3624, 3626, 3628, 3631, 3635, 3641, 3652, 3657, 3660,
       3667, 3674, 3675, 3678, 3679, 3682, 3683, 3684, 3686, 3687, 3701,
       3707, 3708, 3709, 3714, 3719, 3721, 3726, 3727, 3729, 3731, 3733,
       3736, 3739, 3742, 3750, 3752, 3755, 3761, 3765, 3767, 3768, 3774,
       3775, 3777, 3792, 3793, 3795, 3797, 3801, 3802, 3805, 3807, 3811,
       3815, 3816, 3819, 3827, 3830, 3836, 3838, 3840, 3842, 3844, 3845,
       3852, 3854, 3862, 3867, 3869, 3870, 3872, 3876, 3878, 3880, 3884,
       3885, 3886, 3887, 3888, 3889, 3896, 3900, 3903, 3910, 3913, 3919,
       3922, 3928, 3932, 3934, 3937, 3938, 3939, 3946, 3954, 3959, 3960,
       3963, 3965, 3966, 3968, 3973, 3975, 3976, 3981, 3982, 3985, 3986,
       3988, 3995, 3996, 3997, 4003, 4008, 4012, 4016, 4017, 4018, 4020,
       4021, 4022, 4033, 4042, 4043, 4044, 4051, 4054, 4058, 4060, 4070,
       4073, 4081, 4085, 4087, 4102, 4109, 4114, 4117, 4118, 4119, 4122,
       4123, 4128, 4130, 4131, 4132, 4138, 4143, 4144, 4148, 4152, 4157,
       4158, 4164, 4166, 4169, 4170, 4171, 4177, 4185, 4186, 4187, 4189,
       4192, 4194, 4200, 4207, 4209, 4213, 4217, 4220, 4222, 4230, 4234,
       4239, 4247, 4252, 4253, 4255, 4256, 4257, 4258, 4260, 4261, 4265,
       4266, 4270, 4271, 4275, 4280, 4283, 4299, 4304, 4307, 4311, 4319,
       4323, 4326, 4332, 4333, 4337, 4340, 4341, 4342, 4352, 4353, 4363,
       4364, 4365, 4375, 4378, 4383, 4384, 4385, 4390, 4394, 4396, 4398,
       4402, 4412, 4416, 4417, 4418, 4422, 4423, 4425, 4427, 4428, 4429,
       4433, 4442, 4453, 4459, 4462, 4469, 4470, 4476, 4484, 4488, 4489,
       4490, 4491, 4492, 4493, 4494, 4495, 4497, 4498, 4502, 4504, 4511,
       4513, 4517, 4522, 4524, 4532, 4534, 4535, 4536, 4539, 4548, 4550,
       4552, 4554, 4559, 4572, 4573, 4577, 4578, 4579, 4580, 4586, 4592,
       4593, 4600, 4601, 4606, 4609, 4610, 4612, 4615, 4622, 4629, 4630,
       4642, 4643, 4647, 4649, 4651, 4653, 4668, 4671, 4675, 4676, 4677,
       4682, 4684, 4688, 4689, 4692, 4693, 4699, 4705, 4708, 4711, 4717,
       4719, 4722, 4727, 4728, 4730, 4734, 4735, 4736, 4737, 4745, 4746,
       4750, 4752, 4756, 4759, 4762, 4764, 4767, 4770, 4771, 4779, 4782,
       4785, 4790, 4797, 4801, 4803, 4804, 4805, 4810, 4818, 4827, 4829,
       4831, 4837, 4840, 4841, 4843, 4854, 4856, 4858, 4862, 4867, 4871,
       4872, 4876, 4878, 4882, 4885, 4887, 4891, 4896, 4899, 4903, 4904,
       4905, 4907, 4908, 4910, 4915, 4920, 4921, 4924, 4925, 4928, 4931,
       4932, 4934, 4937, 4938, 4940, 4944, 4945, 4949, 4957, 4962, 4965,
       4967, 4970, 4974, 4979, 4981, 4984, 4987, 4993, 4995, 4999]

def get_all_inputs():
    result = {}
    env = get_environment()
    for iname in [f for f in os.listdir('.') if splitext(f)[1] == '.csv']:
        df = env.read_df(iname, index_col=None)
        datasetname, _ = splitext(iname)
        result[datasetname] = df
    return result

def wait_and_save_all_cb(outputs):
    get_environment().store_solution(outputs)

def mp_solution_to_df(solution, mdl_assigned_days, mdl_costs):
    print('***** START')
    assigned_days = np.zeros(NUM_FAMILIES, int)
    for fid, day in mdl_costs:
        if solution[mdl_assigned_days[fid, day]] > 0.5:
            assigned_days[fid] = day + 1
            print("{},{}".format(fid, day+1))

    print('***** STOP')
    solution_df = pd.DataFrame(assigned_days, columns =['assigned_day'])
    solution_df.index.name = 'family_id'
    return solution_df

def build_model(inputs):
    def accounting_penalty(day, next_day):
        return (day-125)*(day**(0.5 + abs(day-next_day)/50.0))/400

    print('Building model')
    family_data = inputs['family_data'].values
    desired = family_data[:,1:11]
    n_people = family_data[:,11]

    starting_solution = inputs['best_submission']
    starting_solution.set_index('family_id', inplace=True)
    starting_solution = starting_solution.values

    solver = Model(name='Santa2019')

    print('Adding costs')
    C = {}
    for fid, choices in enumerate(desired):
        for cid in range(5):
            C[fid, choices[cid]-1] = FAMILY_COST[cid] + n_people[fid] * MEMBER_COST[cid]

    B = solver.binary_var_dict(C, name='B')
    I = solver.integer_var_list(NUM_DAYS, lb=125, ub=300, name='I')

    print('Adding people per day')
    for day in range(NUM_DAYS):
        solver.add(solver.sum(n_people[fid]*B[fid, day] for fid in range(NUM_FAMILIES) if (fid,day) in B) == I[day])

    print('Constraint of one choice per family')
    for fid in range(NUM_FAMILIES):
        solver.add(solver.sum(B[fid, day] for day in range(NUM_DAYS) if (fid,day) in B) == 1)

    print('Preference cost cal')
    preference_cost = solver.sum(C[fid, day]*B[fid, day] for fid, day in B)
    Y = solver.binary_var_cube(NUM_DAYS, 176, 176, name='Y')

    print('Accounting cost cal')
    for day in range(NUM_DAYS):
        next_day = np.clip(day+1, 0, NUM_DAYS-1)
        gen = [(u,v) for v in range(176) for u in range(176)]
        solver.add(solver.sum(Y[day,u,v]*u for u,v in gen) == I[day]-125)
        solver.add(solver.sum(Y[day,u,v]*v for u,v in gen) == I[next_day]-125)
        solver.add(solver.sum(Y[day,u,v]   for u,v in gen) == 1)
        
    gen = [(day,u,v) for day in range(NUM_DAYS) for v in range(176) for u in range(176)]
    accounting_penalties = solver.sum(accounting_penalty(u+125,v+125) * Y[day,u,v] for day,u,v in gen)
    # solver.add(accounting_penalties <= 6300)
    solver.add(accounting_penalties >= 5520)

    solver.minimize(accounting_penalties + preference_cost)

    print('Setting best submission')
    sub = inputs['best_submission']
    sub_cst = solver.add_constraints(B[f, a-1] == 1 for f,a in zip(sub.index, sub.assigned_day))

    solver.parameters.threads = 1
    solver.parameters.mip.tolerances.mipgap = 0.00

    # return solver, B, C

    print('Doing first solve')
    if solver.solve():
        print('Preparing solve has been completed')
        print('Objective sample: {}'.format(solver.objective_value))
        solver.report()

        solver.remove(sub_cst)

        priority_sub = sub.loc[sub.index.isin(priority_fids)]
        solver.add_constraints(B[f, a-1] == 1 for f,a in zip(priority_sub.index, priority_sub.assigned_day))

        solver.parameters.threads = 10
        solver.parameters.timelimit = 60 * 55
        solver.print_information()
        return solver, B, C
    else:
        print('Waiting....')
        time.sleep(10)
        return solver, B, C


    # start = dict()
    # for fid, day in C:
    #     best_day = starting_solution[fid][0] - 1
    #     if best_day == day:
    #         if fid == 0:
    #             print("Test on 0: {}".format(day))
    #         start[B[fid, day]] = 1
    #     else:
    #         if fid == 0:
    #             print("Test on 0. Has value 0 : {}".format(day))
    #         start[B[fid, day]] = 0
    # solver.add_mip_start(SolveSolution(solver, start))
    
    # solver.parameters.timelimit = 60 * 45

    # solver.print_information()

    # return solver, B, C

if __name__ == '__main__':
    inputs = get_all_inputs()
    outputs = {}

    get_environment().abort_callbacks += [partial(wait_and_save_all_cb, outputs)]

    mdl, mdl_assigned_days, mdl_costs = build_model(inputs)

    print('Solving...')
    if not mdl.solve():
        print('*** Problem has no solution')
        # print(mdl.solve_details())
    else:
        print('* model solved as function:')
        print('Objective: {}'.format(mdl.objective_value))
        # mdl.print_solution()
        mdl.report_kpis()
        # Save the CPLEX solution as 'solution.csv' program output
        solution_df = mp_solution_to_df(mdl.solution, mdl_assigned_days, mdl_costs)
        outputs['solution'] = solution_df
        get_environment().store_solution(outputs)
#!/usr/bin/env python


# from numba import njit, jitclass
import numpy as np
import pandas as pd

from tqdm import tqdm_notebook as tqdm
import os
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
import math

family_data_path = './input/family_data.csv'
submission_path = './input/submission-76000.csv'
model_path = './input/reinforcement-workshop.h5'


# In[7]:


class Cost:
    def __init__(self, family_data_path=family_data_path):
        family = pd.read_csv(family_data_path, index_col='family_id')
        self.family_size = family.n_people.values.astype(np.int8)
        self.family_cost_matrix = self._penalty_array(family, self.family_size)
        self.accounting_cost_matrix = self._accounting_cost_matrix()

    @staticmethod
    def _penalty_array(family, family_size):
        penalties = np.asarray([
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
            ] for n in range(family_size.max() + 1)
        ])
        family_cost_matrix = np.concatenate(family.n_people.apply(
            lambda n: np.repeat(penalties[n, 10], 100).reshape(1, 100)))
        for fam in family.index:
            for choice_order, day in enumerate(family.loc[fam].drop("n_people")):
                family_cost_matrix[fam, day - 1] = penalties[family.loc[fam, "n_people"], choice_order]
        return family_cost_matrix

    @staticmethod
    def _accounting_cost_matrix():
        accounting_cost_matrix = np.zeros((1000, 500))
        for n in range(accounting_cost_matrix.shape[0]):
            for diff in range(accounting_cost_matrix.shape[1]):
                accounting_cost_matrix[n, diff] = max(0, (n - 125.0) / 400.0 * n ** (0.5 + diff / 50.0))
        return accounting_cost_matrix

    def calculate(self, prediction):
        p, ac, nl, nh = self._calculate(prediction, self.family_size, self.family_cost_matrix,
                                        self.accounting_cost_matrix)
        return (p + ac) + (nl + nh) * 1000000

    @staticmethod
    #     @njit(fastmath=True)
    def _calculate(prediction, family_size, family_cost_matrix, accounting_cost_matrix):
        N_DAYS = 100
        MAX_OCCUPANCY = 300
        MIN_OCCUPANCY = 125
        penalty = 0
        daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int16)
        for i, (pred, n) in enumerate(zip(prediction, family_size)):
            daily_occupancy[pred - 1] += n
            penalty += family_cost_matrix[i, pred - 1]

        accounting_cost = 0
        n_low = 0
        n_high = 0
        daily_occupancy[-1] = daily_occupancy[-2]
        for day in range(N_DAYS):
            n_next = daily_occupancy[day + 1]
            n = daily_occupancy[day]
            n_high += (n > MAX_OCCUPANCY)
            n_low += (n < MIN_OCCUPANCY)
            diff = abs(n - n_next)
            accounting_cost += accounting_cost_matrix[n, diff]

        return np.asarray([penalty, accounting_cost, n_low, n_high])


# In[8]:


class Workshop:
    def __init__(self):
        self.family_sizes = []
        self.family_choices = []
        self.assigned_days = []
        self.weight_days = []
        self.changes_left = 0
        self.current_family_id = 0
        self.cost = Cost()

    def reset(self, number_moves=1):
        self._set_state()
        self.changes_left = 10000
        self.current_family_id = 0
        return self._get_state(number_moves)

    def step(self, family_index, day_choice_index, number_moves=1):
        self.changes_left -= 1
        self.current_family_id = (self.current_family_id + 1) % 5000
        reward = -self.cost.calculate(self.assigned_days)
        self.weights_days[self.assigned_days[family_index] - 1] -= self.family_sizes[family_index]
        self.assigned_days[family_index] = self.family_choices[family_index, day_choice_index]
        self.weights_days[self.assigned_days[family_index] - 1] += self.family_sizes[family_index]
        state, family_ids = self._get_state(number_moves)
        return state, family_ids, reward, self._is_done()

    def get_submission(self):
        submission = pd.Series(self.assigned_days, name="assigned_day")
        submission.index.name = "family_id"
        score = self.cost.calculate(self.assigned_days)
        return submission, score

    def _set_state(self):
        family = pd.read_csv(family_data_path, index_col='family_id')
        choice_cols = ['choice_{}'.format(i) for i in range(10)]
        self.family_choices = np.array(family[choice_cols])
        self.family_choices_t = np.transpose(np.array(family[choice_cols]))
        self.family_sizes = np.array(family['n_people'])

        submission = pd.read_csv(submission_path, index_col='family_id')
        self.assigned_days = submission['assigned_day'].values
        self.weights_days = np.zeros(100)
        for i in range(len(self.assigned_days)):
            self.weights_days[self.assigned_days[i] - 1] += self.family_sizes[i]

    def _get_state(self, number_moves, shuffle=False):
        if shuffle:
            family_ids = np.random.choice(5000, number_moves, replace=False)
        else:
            family_ids = [self.current_family_id]
        fam_choices = [self.family_choices_t[i] for i in range(10)]
        state = [state_obj[:, None] for state_obj in [self.family_sizes] + fam_choices]

        state = [[state_obj[i] for i in family_ids] for state_obj in state]
        state = [[np.transpose(self.weights_days) for i in range(number_moves)]] + state
        return state, family_ids

    def _is_done(self):
        return any(people_in_day > 300 or people_in_day < 125 for people_in_day in self.weights_days) \
               or self.changes_left < 0


# In[15]:

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.5  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_max = 1.0
        self.lmbda = 0.001
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.tau = 0.08
        self.steps = 0
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def reset_memory(self):
        self.memory = deque(maxlen=2500)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        weight_days = Input(shape=(100,))
        family_sizes = Input(shape=(1,))

        family_choices_0 = Input(shape=(1,))
        family_choices_1 = Input(shape=(1,))
        family_choices_2 = Input(shape=(1,))
        family_choices_3 = Input(shape=(1,))
        family_choices_4 = Input(shape=(1,))
        family_choices_5 = Input(shape=(1,))
        family_choices_6 = Input(shape=(1,))
        family_choices_7 = Input(shape=(1,))
        family_choices_8 = Input(shape=(1,))
        family_choices_9 = Input(shape=(1,))

        a = Dense(128, activation='relu')(weight_days)
        a = Dense(128, activation='relu')(a)
        b = Dense(128, activation='relu')(family_sizes)
        c = Concatenate()([family_choices_0, family_choices_1, family_choices_2, family_choices_3, family_choices_4,
                           family_choices_5, family_choices_6, family_choices_7, family_choices_8, family_choices_9])
        # c = Flatten()(family_choices)
        c = Dense(128, activation='relu')(c)

        x = Concatenate()([a, b, c])
        x = Dense(64, activation='relu')(x)
        x = Dense(10, activation='softmax')(x)

        model = Model([weight_days, family_sizes, family_choices_0, family_choices_1, family_choices_2,
                       family_choices_3, family_choices_4, family_choices_5, family_choices_6, family_choices_7,
                       family_choices_8, family_choices_9], [x])

        model.compile(loss=tf.keras.losses.mean_squared_error,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.steps += 1
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, family_ids):
        if np.random.rand() <= self.epsilon:
            days = np.random.randint(0, 9, len(family_ids))
        else:
            act_values = self.model.predict(state)
            days = act_values.argmax(axis=1)
        return np.transpose([family_ids, days])

    def replay(self, batch_size):
        if len(self.memory) < batch_size * 3:
            return 0
        batch = random.sample(self.memory, batch_size)
        states = self.extract_states([val[0] for val in batch], batch_size)
        actions = np.array([val[1][1] for val in batch])
        rewards = np.array([val[2] for val in batch])
        next_states = self.extract_states([val[3] for val in batch], batch_size)

        # predict Q(s,a) given the batch of states
        target_q = self.model.predict(states)
        # predict Q(s',a') from the evaluation network
        prim_qtp1 = self.model.predict(next_states)
        # copy the prim_qt tensor into the target_q tensor - we then will update one index corresponding to the max action
        updates = rewards
        valid_idxs = np.array(next_states).sum(axis=1) != 0
        batch_idxs = np.arange(batch_size)
        if self.target_model is None:
            updates[valid_idxs] += self.gamma * np.amax(prim_qtp1[valid_idxs, :], axis=1)
        else:
            prim_action_tp1 = np.argmax(prim_qtp1, axis=1)
            q_from_target = self.target_model.predict(next_states)
            updates[valid_idxs] += self.gamma * q_from_target[batch_idxs[valid_idxs], prim_action_tp1[valid_idxs]]
        target_q[batch_idxs, actions] = updates
        loss = self.model.train_on_batch(states, target_q)

        # update target network parameters slowly from primary network
        for t, e in zip(self.target_model.trainable_variables, self.model.trainable_variables):
            t.assign(t * (1 - self.tau) + e * self.tau)

        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.lmbda * self.steps)
        return loss

    def extract_states(self, pre_states, batch_size):
        for j in range(batch_size):
            if pre_states[j] is None:
                pre_states[j] = [np.array([i]) for i in np.zeros(12)]
                pre_states[j][0] = [np.zeros(100)]
        states = np.transpose(pre_states)[0]
        for i in range(1, 12):
            for j in range(batch_size):
                if isinstance(type(states[i][j]), type(np.array(0))):
                    states[i][j] = states[i][j][0]
                else:
                    states[i][j] = 0.0
        states = states.tolist()
        return states

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# In[16]:

def run(episodes, load_model=False):
    env = Workshop()
    agent = DQNAgent()
    if load_model:
        agent.load(model_path)
    done = False
    batch_size = 8

    for e in range(episodes):
        state, family_ids = env.reset()
        #         agent.reset_memory()
        cnt = 0
        avg_loss = 0

        while True:
            actions = agent.act(state, family_ids)
            for family_id, day in actions:
                next_state, family_ids, reward, done = env.step(family_id, day)
                if done:
                    next_state = None
                agent.remember(state, [family_id, day], reward, next_state, done)
                state = next_state
            loss = agent.replay(batch_size)
            avg_loss += loss
            if done:
                avg_loss /= cnt
                print("episode: {}/{}, changes done: {}, score: {}, avg_loss= {:.3}, e: {:.2}"
                      .format(e, episodes, cnt, reward, avg_loss, agent.epsilon))
                break
            cnt += 1
        print("Episode {} over. Reward finished on: {}".format(e, reward))
        agent.save("./reinforcement-workshop.h5")
        # if e % 10 == 0:
        #     agent = DQNAgent()
        #     agent.load("./reinforcement-workshop.h5")


run(1000)

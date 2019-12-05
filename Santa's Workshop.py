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

family_data_path = './input/family_data.csv'
submission_path = './input/submission-76000.csv'


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
        self.changes_left = 0
        self.cost = Cost()

    def reset(self, number_moves=1):
        self._set_state()
        self.changes_left = 10000
        return self._get_state(number_moves)

    def step(self, family_index, day_choice_index, number_moves=1):
        self.changes_left -= 1
        reward = -self.cost.calculate(self.assigned_days)
        self.assigned_days[family_index] = self.family_choices[family_index, day_choice_index]
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

    def _get_state(self, number_moves):
        family_ids = np.random.choice(5000, number_moves, replace=False)
        fam_choices = [self.family_choices_t[i] for i in range(10)]
        state = [state_obj[:, None] for state_obj in [self.family_sizes] + fam_choices]
        state = [[np.transpose(self.assigned_days) for i in range(5000)]] + state

        state = [[state_obj[i] for i in family_ids] for state_obj in state]
        return state, family_ids
        # return [[np.transpose(self.assigned_days)], np.transpose(self.family_sizes), np.transpose(self.family_choices_t[0]),
        #         np.transpose(self.family_choices_t[1]), np.transpose(self.family_choices_t[2]),
        #         np.transpose(self.family_choices_t[3]), np.transpose(self.family_choices_t[4]),
        #         np.transpose(self.family_choices_t[5]), np.transpose(self.family_choices_t[6]),
        #         np.transpose(self.family_choices_t[7]), np.transpose(self.family_choices_t[8]),
        #         np.transpose(self.family_choices_t[9])]

    def _is_done(self):
        return False  # self.changes_left < 0


# In[15]:


class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = tf.abs(error) <= clip_delta

        squared_loss = 0.5 * tf.square(error)
        quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)

        return tf.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        assigned_days = Input(shape=(5000,))
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

        a = Dense(512, activation='relu')(assigned_days)
        a = Dense(256, activation='relu')(a)
        b = Dense(256, activation='relu')(family_sizes)
        c = Concatenate()([family_choices_0, family_choices_1, family_choices_2, family_choices_3, family_choices_4,
                           family_choices_5, family_choices_6, family_choices_7, family_choices_8, family_choices_9])
        # c = Flatten()(family_choices)
        c = Dense(256, activation='relu')(c)

        x = Concatenate()([a, b, c])
        x = Dense(256, activation='relu')(x)
        x = Dense(10, activation='softmax')(x)

        model = Model([assigned_days, family_sizes, family_choices_0, family_choices_1, family_choices_2,
                       family_choices_3, family_choices_4, family_choices_5, family_choices_6, family_choices_7,
                       family_choices_8, family_choices_9], [x])

        model.compile(loss=tf.losses.mean_squared_error,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, family_ids):
        if np.random.rand() <= self.epsilon:
            days = np.random.randint(0, 9, len(family_ids))
        else:
            act_values = self.model.predict(state)
            days = act_values.argmax(axis=1)
        return np.transpose([family_ids, days])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            action_family = action[0]
            action_day = action[1]
            target = self.model.predict(state)
            if done:
                target[0][action_day] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action_day] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# In[16]:

def run(episodes, load_model=False):
    env = Workshop()
    agent = DQNAgent()
    # agent.load("./reinforcement-workshop.h5")
    done = False
    batch_size = 8

    for e in range(episodes):
        state, family_ids = env.reset()
        for i in range(5000):
            actions = agent.act(state, family_ids)
            for family_id, day in actions:
                next_state, family_ids, reward, done = env.step(family_id, day)
                agent.remember(state, [family_id, day], reward, next_state, done)
                state = next_state
            if len(agent.memory) % batch_size == 0:
                agent.replay(batch_size)
                print("episode: {}/{}, changes left: {}, score: {}, e: {:.2}"
                      .format(e, episodes, env.changes_left, reward, agent.epsilon))
        agent.update_target_model()
        agent.save("./reinforcement-workshop.h5")


run(10)

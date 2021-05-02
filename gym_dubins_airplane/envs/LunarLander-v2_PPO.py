import os
import gym
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import tensorflow as tf
import gym_dubins_airplane
import numpy as np
import copy
from time import sleep
from utils import rearrangeticks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from multiprocessing import Process, Pipe

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.compat.v1.disable_eager_execution()

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass


class Environment(Process):
    def __init__(self,
                 env_idx,
                 child_conn,
                 env_name,
                 state_size,
                 action_size,
                 visualize=False):
        super(Environment, self).__init__()
        self.env = gym.make(env_name)
        self.is_render = visualize
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.state_size = state_size
        self.action_size = action_size

    def run(self):
        super(Environment, self).run()
        state = self.env.reset()
        self.child_conn.send(state)
        while True:
            action = self.child_conn.recv()
            if self.is_render and self.env_idx == 0:
                self.env.render()

            state, reward, done, info = self.env.step(action)

            if done:
                state = self.env.reset()

            self.child_conn.send([state, reward, done, info])


class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space
        X = Dense(256,
                  activation="relu",
                  kernel_initializer=tf.random_normal_initializer(
                      stddev=0.01))(X_input)
        X = Dense(
            128,
            activation="relu",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(
            64,
            activation="relu",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_space, activation="softmax")(X)
        self.Actor = Model(inputs=X_input, outputs=output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))

    def ppo_loss(self, y_true, y_pred):
        advantages = y_true[:, :1]
        prediction_picks = y_true[:, 1:1 + self.action_space]
        actions = y_true[:, 1 + self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        prob = actions * y_pred
        old_prob = actions * prediction_picks
        prob = np.clip(prob, 1e-10, 1.0)
        old_prob = np.clip(old_prob, 1e-10, 1.0)
        ratio = np.exp(np.log(prob) - np.log(old_prob))
        p1 = ratio * advantages
        p2 = np.clip(ratio, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING) * advantages
        actor_loss = -np.mean(np.minimum(p1, p2))
        entropy = -(y_pred * np.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * np.mean(entropy)
        total_loss = actor_loss - entropy
        return total_loss

    def predict(self, state):
        return self.Actor.predict(state)


class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        old_values = Input(shape=(1, ))
        V = Dense(256, activation="relu",
                  kernel_initializer='he_uniform')(X_input)
        V = Dense(128, activation="relu", kernel_initializer='he_uniform')(V)
        V = Dense(64, activation="relu", kernel_initializer='he_uniform')(V)
        value = Dense(1, activation=None)(V)
        self.Critic = Model(inputs=[X_input, old_values], outputs=value)
        self.Critic.compile(loss=self.critic_PPO2_loss(old_values),
                            optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + np.clip(
                y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss)**2
            v_loss2 = (y_true - y_pred)**2
            value_loss = 0.5 * np.mean(np.maximum(v_loss1, v_loss2))
            return value_loss

        return loss

    def predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])


class PPOAgent:
    def __init__(self, env_name, alg_name="PPO"):
        self.env_name = env_name
        self.algorithm = alg_name
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 5000
        self.episode = 0
        self.max_average = 0  # when average score is above 0 model will be saved
        self.lr = 5e-4  # Org. 0.00025
        self.lr_list = []
        self.lr_list.append(self.lr)
        self.episodes_lr = []
        self.episodes_lr.append(0)
        self.epochs = 10  # training epochs
        self.shuffle = False
        self.Training_batch = 500  # Org. 1000
        self.optimizer = Adam
        self.scores, self.episodes, self.average, self.episodes_lr, self.lr_list = [], [], [], [], []
        self.Actor = Actor_Model(input_shape=self.state_size,
                                 action_space=self.action_size,
                                 lr=self.lr,
                                 optimizer=self.optimizer)
        self.Actor_name = f"{self.env_name}_PPO_Actor.h5"
        self.Critic = Critic_Model(input_shape=self.state_size,
                                   action_space=self.action_size,
                                   lr=self.lr,
                                   optimizer=self.optimizer)
        self.Critic_name = f"{self.env_name}_PPO_Critic.h5"

    def act(self, state):
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot, prediction

    def get_gaes(self,
                 rewards,
                 dones,
                 values,
                 next_values,
                 gamma=0.99,
                 lamda=0.9,
                 normalize=True):
        deltas = [
            r + gamma * (1 - d) * nv - v
            for r, d, nv, v in zip(rewards, dones, next_values, values)
        ]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones,
               next_states):
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values),
                                           np.squeeze(next_values))
        self.Actor.Actor.fit(states,
                             np.hstack([advantages, predictions, actions]),
                             epochs=self.epochs,
                             verbose=0,
                             shuffle=self.shuffle)
        self.Critic.Critic.fit([states, values],
                               target,
                               epochs=self.epochs,
                               verbose=0,
                               shuffle=self.shuffle)

    def load(self):
        self.Actor.Actor.load_weights(self.Actor_name)
        self.Critic.Critic.load_weights(self.Critic_name)

    def save(self):
        self.Actor.Actor.save_weights(self.Actor_name)
        self.Critic.Critic.save_weights(self.Critic_name)

    def PlotModel(self, score, episode, FINISHED=False):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-100:]) / 100)
        # much faster than episode % 100
        if FINISHED:
            self.scores = np.array(self.scores)
            self.episodes = np.array(self.episodes)
            fig = plt.figure(constrained_layout=True)
            gs0 = gs.GridSpec(1, 2, figure=fig)
            ax1 = fig.add_subplot(gs0[0])
            ax1.plot(self.episodes, self.scores, 'b')
            ax1.plot(self.episodes, self.average, 'r')
            ax1.set_title("Training cycle")
            ax1.set_ylabel('Score')
            ax1.set_xlabel('Steps')
            # hl_y = [*zip(self.scores_[self.scores_ > self.average_], self.episodes_[self.scores_ > self.average_])]
            # hl_x = [*zip(self.scores_[self.scores_ < self.average_], self.episodes_[self.scores_ < self.average_])]
            rearrangeticks(ax1, 0, episode, min(self.scores), max(self.scores),
                           7, 7)
            y_err = self.episodes.std() * np.sqrt(
                1 / len(self.episodes) +
                (self.episodes - self.episodes.mean())**2 / np.sum(
                    (self.episodes - self.episodes.mean())**1))
            ax1.fill_between(self.episodes,
                             self.average - y_err,
                             self.average + y_err,
                             alpha=0.2)
            # ConstraintPoly(ax1, self.scores_[self.scores_ > self.average_], self.episodes_[self.scores_ > self.average_], "red", .2)
            # ax1.add_patch(Polygon([
            #     *zip(
            #         self.scores_[self.scores_ < self.average_],
            #         self.scores_[self.scores_ > self.average_])],
            #     color="red", alpha=.2))
            ax2 = plt.subplot(gs0[1])
            ax2.plot(self.episodes_lr, self.lr_list)
            ax2.set_title("Learning rate decay")
            ax2.set_ylabel('Learning rate')
            ax2.set_xlabel('Steps')
            rearrangeticks(ax2, min(self.episodes_lr), max(self.episodes_lr),
                           min(self.lr_list), max(self.lr_list), 7, 7)
            fig.suptitle(self.env_name + ' ' + self.algorithm)
            plt.savefig(self.env_name + ".png", bbox_inches='tight')
            plt.show()
        if self.average[-1] >= self.max_average:
            self.max_average = self.average[-1]
            self.save()
            # decreaate learning rate every saved model
            self.lr *= 0.995
            self.lr_list.append(self.lr)
            self.episodes_lr.append(episode)
            K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)
        return self.average[-1]

    def run_batch(self):  # train every self.Training_batch episodes
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        # scores_window = deque(maxlen=100)
        done, score = False, 0
        average = 0
        while True:
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            for t in range(self.Training_batch):
                self.env.render()
                # sleep(.2)
                action, action_onehot, prediction = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                states.append(state)
                next_states.append(
                    np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                # Update current state
                state = next_state
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                # scores_window.append(score)
                if done:
                    self.episode += 1
                    average = self.PlotModel(score, self.episode)
                    if self.episode % 50 == 0:
                        print("episode: {}/{}, score: {}, average: {:.2f}".
                              format(self.episode, self.EPISODES, score,
                                     average))
                    state, done, score = self.env.reset(), False, 0
                    state = np.reshape(state, [1, self.state_size[0]])
                    print("600")
                    break
            self.replay(states, actions, rewards, predictions, dones,
                        next_states)
            if average >= 300:
                print(
                    '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
                    .format(self.episode, average))
                break
            if self.episode >= self.EPISODES:
                break
        self.env.close()

    def run_multiprocesses(self, num_worker=8):
        works, parent_conns, child_conns = [], [], []
        for idx in range(num_worker):
            parent_conn, child_conn = Pipe()
            work = Environment(idx, child_conn, self.env_name,
                               self.state_size[0], self.action_size, False)
            work.start()
            works.append(work)
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)
        states = [[] for _ in range(num_worker)]
        next_states = [[] for _ in range(num_worker)]
        actions = [[] for _ in range(num_worker)]
        rewards = [[] for _ in range(num_worker)]
        dones = [[] for _ in range(num_worker)]
        predictions = [[] for _ in range(num_worker)]
        score = [0 for _ in range(num_worker)]
        state = [0 for _ in range(num_worker)]
        for worker_id, parent_conn in enumerate(parent_conns):
            state[worker_id] = parent_conn.recv()
        average = 0
        while self.episode < self.EPISODES:
            predictions_list = self.Actor.predict(
                np.reshape(state, [num_worker, self.state_size[0]]))
            actions_list = [
                np.random.choice(self.action_size, p=i)
                for i in predictions_list
            ]
            for worker_id, parent_conn in enumerate(parent_conns):
                parent_conn.send(actions_list[worker_id])
                action_onehot = np.zeros([self.action_size])
                action_onehot[actions_list[worker_id]] = 1
                actions[worker_id].append(action_onehot)
                predictions[worker_id].append(predictions_list[worker_id])
            for worker_id, parent_conn in enumerate(parent_conns):
                next_state, reward, done, _ = parent_conn.recv()
                states[worker_id].append(state[worker_id])
                next_states[worker_id].append(next_state)
                rewards[worker_id].append(reward)
                dones[worker_id].append(done)
                state[worker_id] = next_state
                score[worker_id] += reward
                if done:
                    average = self.PlotModel(score[worker_id], self.episode)
                    if self.episode % 50 == 0:
                        print(
                            "episode: {}/{}, worker: {}, score: {}, average: {:.2f}"
                            .format(self.episode, self.EPISODES, worker_id,
                                    score[worker_id], average))
                    score[worker_id] = 0
                    if (self.episode < self.EPISODES):
                        self.episode += 1
            for worker_id in range(num_worker):
                if len(states[worker_id]) >= self.Training_batch:
                    self.replay(states[worker_id], actions[worker_id],
                                rewards[worker_id], predictions[worker_id],
                                dones[worker_id], next_states[worker_id])
                    states[worker_id] = []
                    next_states[worker_id] = []
                    actions[worker_id] = []
                    rewards[worker_id] = []
                    dones[worker_id] = []
                    predictions[worker_id] = []
            if average >= 300:
                self.PlotModel(score[worker_id], self.episode, True)
                print(
                    "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}"
                    .format(self.episode, average))
                break
        # terminating processes after while loop
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()

    def test(self, test_episodes=100):
        self.load()
        for e in range(100):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax(self.Actor.predict(state)[0])
                state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(
                        e, test_episodes, score))
                    break
        self.env.close()

    def test_multiagent(self, test_episodes=100):
        self.load()
        for e in range(100):
            state, state_red = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            state_red = np.reshape(state_red, [1, self.state_size[0]])
            done = False
            done_red = False
            score = 0
            score_red = 0
            for t in range(self.Training_batch):
                self.env.render()
                sleep(0.01)
                action = np.argmax(self.Actor.predict(state)[0])
                actionred = np.argmax(self.Actor.predict(state_red)[0])
                state, reward, done, info, state_red, reward_red, done_red, info_red = self.env.step(
                    action, actionred)
                state = np.reshape(state, [1, self.state_size[0]])
                state_red = np.reshape(state_red, [1, self.state_size[0]])
                score += reward
                score_red += reward_red
                if t == self.Training_batch - 1:
                    done = True
                if done or done_red:
                    print("episode: {}/{}, score: {}".format(
                        e, test_episodes, score))
                    break
        self.env.close()


if __name__ == "__main__":
    env_name = 'dubinsAC2D-v0'
    agent = PPOAgent(env_name)
    # agent.run()
    # agent.run_batch()
    # agent.run_multiprocesses()
    agent.test()

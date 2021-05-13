import os
import gym
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import tensorflow as tf
import gym_dubins_airplane
import numpy as np
import copy
import argparse
from time import sleep
from utils import rearrangeticks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from config import Config
from multiprocessing import Process

tf.compat.v1.disable_eager_execution()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # -1:cpu, 0:first gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To disable TensorFlow logs

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass


def get_args():
    parser = argparse.ArgumentParser(
        description="Following arguments are available.")
    parser.add_argument("-r",
                        "--render",
                        action="store_true",
                        help="rendering type")
    parser.add_argument("-s", "--slowdown", type=str, help="Speed buff")
    parser.add_argument("-t",
                        "--test",
                        action="store_true",
                        help="Test the environment")
    return parser.parse_args()


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
        state = np.reshape(state, [1, self.state_size])
        self.child_conn.send(state)
        while True:
            action = self.child_conn.recv()
            if self.is_render and self.env_idx == 0:
                self.env.render()

            state, reward, done, info = self.env.step(action)
            state = np.reshape(state, [1, self.state_size])

            if done:
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

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
            512,
            activation="relu",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(
            256,
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
        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)
        ratio = K.exp(K.log(prob) - K.log(old_prob))
        p1 = ratio * advantages
        p2 = K.clip(ratio,
                    min_value=1 - LOSS_CLIPPING,
                    max_value=1 + LOSS_CLIPPING) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        total_loss = actor_loss - entropy
        return total_loss

    # def ppo_loss(self, y_true, y_pred):
    #     advantages = y_true[:, :1]
    #     prediction_picks = y_true[:, 1:1 + self.action_space]
    #     actions = y_true[:, 1 + self.action_space:]
    #     LOSS_CLIPPING = 0.2
    #     ENTROPY_LOSS = 0.001
    #     prob = actions * y_pred
    #     old_prob = actions * prediction_picks
    #     prob = np.clip(prob, 1e-10, 1.0)
    #     old_prob = np.clip(old_prob, 1e-10, 1.0)
    #     ratio = np.exp(np.log(prob) - np.log(old_prob))
    #     p1 = ratio * advantages
    #     p2 = np.clip(ratio, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING) * advantages
    #     actor_loss = -np.mean(np.minimum(p1, p2))
    #     entropy = -(y_pred * np.log(y_pred + 1e-10))
    #     entropy = ENTROPY_LOSS * np.mean(entropy)
    #     total_loss = actor_loss - entropy
    #     return total_loss

    def predict(self, state):
        return self.Actor.predict(state)


class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        old_values = Input(shape=(1, ))
        V = Dense(256, activation="relu",
                  kernel_initializer='he_uniform')(X_input)
        V = Dense(512, activation="relu", kernel_initializer='he_uniform')(V)
        V = Dense(256, activation="relu", kernel_initializer='he_uniform')(V)
        value = Dense(1, activation=None)(V)
        self.Critic = Model(inputs=[X_input, old_values], outputs=value)
        self.Critic.compile(loss=[self.critic_PPO2_loss(old_values)],
                            optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + K.clip(y_pred - values,
                                                 -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss)**2
            v_loss2 = (y_true - y_pred)**2
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            return value_loss

        return loss

    def predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])


class PPOAgent:
    def __init__(self, env_name, alg_name="PPO"):
        self.env_name = env_name
        self.algorithm = alg_name
        self.env = gym.make(env_name)
        self.action_size = Config.action_size
        self.state_size = 17
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
        self.Training_batch = 600  # Org. 1000
        self.optimizer = Adam
        self.scores, self.episodes, self.average, self.episodes_lr, self.lr_list = [], [], [], [], []
        self.Actor = Actor_Model(input_shape=self.state_size,
                                 action_space=self.action_size,
                                 lr=self.lr,
                                 optimizer=self.optimizer)
        self.Actor_name = "actor.h5"
        self.Critic = Critic_Model(input_shape=self.state_size,
                                   action_space=self.action_size,
                                   lr=self.lr,
                                   optimizer=self.optimizer)
        self.Critic_name = "critic.h5"

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

    def load(self, actor_name, critic_name):
        Actor = Actor_Model(input_shape=self.state_size,
                            action_space=self.action_size,
                            lr=5e-4,
                            optimizer=self.optimizer)
        Critic = Critic_Model(input_shape=self.state_size,
                              action_space=self.action_size,
                              lr=5e-4,
                              optimizer=self.optimizer)
        Actor.Actor.load_weights(actor_name)
        Critic.Critic.load_weights(critic_name)
        return Actor, Critic

    def load_v0(self):
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
                    (self.episodes - self.episodes.mean())**2))
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
            self.lr *= 0.994
            self.lr_list.append(self.lr)
            self.episodes_lr.append(episode)
            K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)
        return self.average[-1]

    def tac_memory(self, t):
        for idx, veh in enumerate((self.env._blueAC, self.env._redAC)):
            pos, _, att, _ = veh.get_sta()
            att[1] *= -1
            att[0] *= -1
            if idx == 0:
                self.tacview_blue = np.vstack(
                    (self.tacview_blue, (t * .12, *pos, *att)))
            else:
                self.tacview_red = np.vstack(
                    (self.tacview_red, (t * .12, *pos, *att)))

    def run_batch(self):
        done, score, _, infos, average = False, 0, '', [], 0
        self.info_list = ["win", "tie", "loss", "collision"]
        args = get_args()
        for episode in range(self.EPISODES):
            state = self.env.reset()
            self.tacview_red = np.zeros(7)
            self.tacview_blue = np.zeros(7)
            state = np.reshape(state, [1, self.state_size])
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            info = "tie"
            for t in range(self.Training_batch):
                if args.render:
                    self.env.render()
                if args.slowdown == "slow":
                    sleep(.1)
                elif args.slowdown == "fast":
                    sleep(.025)
                action, action_onehot, prediction = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                self.tac_memory(t)
                states.append(state)
                next_states.append(np.reshape(next_state,
                                              [1, self.state_size]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                state = np.reshape(next_state, [1, self.state_size])
                score += reward
                if done or t == self.Training_batch - 1:
                    np.savetxt(
                        "red.csv",
                        self.tacview_blue[1:],
                        fmt="%3.4f",
                        delimiter=',',
                        header=
                        "Time,Longitude,Latitude,Altitude,Roll,Pitch,Yaw",
                        comments='')
                    np.savetxt(
                        "blue.csv",
                        self.tacview_red,
                        fmt="%3.4f",
                        delimiter=',',
                        header=
                        "Time,Longitude,Latitude,Altitude,Roll,Pitch,Yaw",
                        comments='')
                    infos.append(info)
                    average = self.PlotModel(score, episode)
                    if self.episode % 100 == 0:
                        print(
                            f"episode: {episode}, score: {score}, average: {average:.{2}}"
                        )
                    state, done, score, _ = self.env.reset(), False, 0, ''
                    state = np.reshape(state, [1, self.state_size])
                    break
            self.replay(states, actions, rewards, predictions, dones,
                        next_states)
        self.env.close()

    def run_batch_multi(self):
        a_1, c_1 = self.load("a_1.h5", "c_1.h5")  # a: actor, c: critic (Enemy)
        state, state_red = self.env.reset()
        # scores_window = deque(maxlen=100)
        state = np.reshape(state, [1, self.state_size[0]])
        state_red = np.reshape(state_red, [1, self.state_size[0]])
        done, score, SAVING, done_red, score_red, average, infos = False, 0, '', False, 0, 0, []
        self.info_list = ["win", "tie", "loss", "collision"]
        while 1:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones, t, _ = [], [], [], [], [], [], 0, []
            while 1:
                # Actor picks an action
                actionred = np.argmax(a_1.predict(state_red)[0])
                action, action_onehot, prediction = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, info, state_red, reward_red, done_red, info_red = self.env.step(
                    action, actionred)
                # Memorize (state, action, reward) for training
                states.append(state)
                next_states.append(
                    np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                # Update current state
                state = np.reshape(next_state, [1, self.state_size[0]])
                state_red = np.reshape(state_red, [1, self.state_size[0]])
                score += reward
                score_red += reward_red
                t += 1
                if t == self.Training_batch:
                    done = True
                    info = "tie"
                if done:
                    infos.append(info)
                    self.episode += 1
                    average, SAVING = self.PlotModel(score, self.episode)
                    print(
                        "episode: {}/{}, score: {}, average: {:.2f} {}".format(
                            self.episode, self.EPISODES, score, average,
                            SAVING))
                    self.writer.add_scalar(f'Workers:{1}/score_per_episode',
                                           score, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/learning_rate',
                                           self.lr, self.episode)

                    state, state_red = self.env.reset()
                    done, score, SAVING, _, score_red = False, 0, '', False, 0
                    state = np.reshape(state, [1, self.state_size[0]])
                    state_red = np.reshape(state_red, [1, self.state_size[0]])

            self.replay(states, actions, rewards, predictions, dones,
                        next_states)
            if self.episode >= self.EPISODES:
                break

        # f = open("run_batch_multi_agent_training.out", 'w')
        # sys.stdout = f
        # ratios = [infos.count(t) / self.EPISODES * 100 for t in self.info_list]
        # res = {
        #     self.info_list[i]: ratios[i]
        #     for i in range(len(self.info_list))
        # }
        # print(res)
        # f.close()
        self.env.close()

    def test(self, test_episodes=100):  # Single agent test
        self.load_v0()
        e = 0  # Episode
        while 1:  # run until test ends
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            score = 0
            # actions = ["acc", "decc", "right", "left"] # Available actions
            while 1:  # runs until done becomes true
                self.env.render()
                action = np.argmax(self.Actor.predict(state))
                # Check if action is acc or decc
                state, reward, done, info = self.env.step(action)
                state = np.reshape(state, [1, self.state_size])
                score += reward
                # done = False
                if done:
                    print(info)
                    print("episode: {}/{}, score: {}".format(
                        e, test_episodes, score))
                    break
            e += 1  # step to next episode in outer while loop
            if e == test_episodes:  # break outer while loop if max episode reached
                break
        self.env.close()

    def test_multi(self, test_episodes=100):  # Multi agent test
        """ This method enables duel of two different agents with different actors and critics.
      To begin with dual agent testing load their corresponding .h5 files.
      To test an agent against itself make sure to load same .h5 files again."""

        actor_1, critic_1 = self.load(
            "a_1.h5", "c_1.h5")  # loads actor and critic of first agent
        actor_2, critic_2 = self.load(
            "a_2.h5", "c_2.h5")  # loads actor and critic of second agent
        e = 0
        while 1:
            state, state_red = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            state_red = np.reshape(state_red, [1, self.state_size[0]])
            done = False
            done_red = False
            score = 0
            score_red = 0
            while 1:
                self.env.render()
                action = np.argmax(actor_1.predict(state)[0])
                action_red = np.argmax(actor_2.predict(state_red)[0])

                state, reward, done, info, state_red, reward_red, done_red, info_red = self.env.step(
                    action, action_red)

                state = np.reshape(state, [1, self.state_size[0]])
                state_red = np.reshape(state_red, [1, self.state_size[0]])
                score += reward
                score_red += reward_red
                if done:
                    print(info)
                    print("episode: {}/{}, score: {}".format(
                        e, test_episodes, score))
                    break
            e += 1
            if e == test_episodes:  # break outer while loop if max episode reached
                break
        self.env.close()


if __name__ == "__main__":
    args = get_args()
    env_name = 'dubinsAC-v0'
    agent = PPOAgent(env_name)
    if args.test:
        agent.test()
    else:
        agent.run_batch()  # train as PPO, train every batch, trains better
    # agent.run_batch_multi() # trains PPO in offline learning against PPO
    # agent.test_multi() # test for multiple agents

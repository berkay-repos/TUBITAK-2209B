import os
import pyglet
import gym
import numpy as np
from math import cos, sin
from random import random
from gym.envs.classic_control import rendering
from config import Config
from ACEnvironment import ACEnvironment2D
from act_cmd import action2command

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


class DubinsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.viewer = None
        self._vel_mps = Config.vel_mps
        self.hl_switch = -2
        self.window_width = Config.window_width
        self.window_height = Config.window_height
        self.window_z = Config.window_z
        self.d_min = Config.d_min
        self.d_max = Config.d_max
        self.blue_health = Config.blue_health
        self.red_health = Config.red_health
        self.n_act = Config.n_act

    def step(self, action):
        u = action2command(self._blueAC._vel_mps, action)
        self._blueAC.takeaction(u[0], u[1], u[2])
        self._redAC.takeaction(self._vel_mps, 0, 0)

        # Red reflect
        if self._redAC._pos_m[0] > self.window_width:
            self._redAC._att[2] = np.pi - np.mod(self._redAC._att[2],
                                                 2 * np.pi)
        elif self._redAC._pos_m[0] < 0:
            self._redAC._att[2] = np.pi - np.mod(self._redAC._att[2],
                                                 2 * np.pi)
        if self._redAC._pos_m[1] > self.window_height:
            self._redAC._att[2] = -np.mod(self._redAC._att[2], 2 * np.pi)
        elif self._redAC._pos_m[1] < 0:
            self._redAC._att[2] = -np.mod(self._redAC._att[2], 2 * np.pi)

        # Blue reflect
        if self._blueAC._pos_m[0] > self.window_width:
            self._blueAC._att[1] = np.pi - np.mod(self._blueAC._att[1],
                                                  2 * np.pi)
        elif self._blueAC._pos_m[0] < 0:
            self._blueAC._att[2] = np.pi - np.mod(self._blueAC._att[2],
                                                  2 * np.pi)
        if self._blueAC._pos_m[1] > self.window_height:
            self._blueAC._att[2] = -np.mod(self._blueAC._att[2], 2 * np.pi)
        elif self._blueAC._pos_m[1] < 0:
            self._blueAC._att[2] = -np.mod(self._blueAC._att[2], 2 * np.pi)

        envSta = self.make_state(self._blueAC, self._redAC)
        reward, terminal, info = self.scalar_reward_terminal()
        return envSta, reward, terminal, info

    def reset(self):
        """Returns randomized position and attitude"""
        self.hl_switch = 0
        self.blue_health = 0
        self.red_health = 0
        self._redAC = ACEnvironment2D(position=np.random.rand(3) * 800,
                                      att=[0, 0, random() * 2 * np.pi],
                                      vel_mps=self._vel_mps)
        self._blueAC = ACEnvironment2D(position=np.random.rand(3) * 800,
                                       att=[0, 0, random() * 2 * np.pi],
                                       vel_mps=self._vel_mps)
        return self.make_state(self._blueAC, self._redAC)

    def render(self, mode='human', close='False'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.window_width,
                                           self.window_height)
            display = pyglet.canvas.Display()
            screen = display.get_default_screen()
            screen_width = screen.width
            screen_height = screen.height
            self.viewer.window.set_location(
                (screen_width - self.window_width) // 2,
                (screen_height - self.window_height) // 2)
            self.viewer.set_bounds(0, self.window_width, 0, self.window_height)

            # Key presses
            @self.viewer.window.event
            def on_key_press(symbol, modifiers):
                if symbol == pyglet.window.key.Q:
                    self.viewer.close()
                if symbol == pyglet.window.key.UP:
                    self._vel_mps += 1
                if symbol == pyglet.window.key.DOWN:
                    self._vel_mps -= 1
                if symbol == pyglet.window.key.UP and modifiers & pyglet.window.key.MOD_SHIFT:
                    self._redAC._att[1] += .2
                if symbol == pyglet.window.key.DOWN and modifiers & pyglet.window.key.MOD_SHIFT:
                    self._redAC._att[1] -= .2
                if symbol == pyglet.window.key.RIGHT:
                    self._redAC._att[0] += .4
                if symbol == pyglet.window.key.LEFT:
                    self._redAC._att[0] -= .4

        # Grid
        ystep = 5
        xstep = 5
        for foo in np.linspace(0, self.window_height, ystep * 5 + 1):
            self.viewer.draw_line((0, foo), (self.window_width, foo),
                                  color=(.8, .8, .8))
        for foo in np.linspace(0, self.window_width, xstep * 5 + 1):
            self.viewer.draw_line((foo, 0), (foo, self.window_height),
                                  color=(.8, .8, .8))
        for foo in np.linspace(0, self.window_height, ystep + 1):
            test = self.viewer.draw_line((0, foo), (self.window_width, foo))
            test.linewidth.stroke = 2
        for foo in np.linspace(0, self.window_width, xstep + 1):
            test = self.viewer.draw_line((foo, 0), (foo, self.window_height))
            test.linewidth.stroke = 2

        # Red aircraft
        dpos, _, datt_rad, dpos_hist = self._redAC.get_sta()
        red_ac_img = rendering.Image('envs/images/f16_red.png', 48, 48)
        red_ac_img._color.vec4 = (1, 1, 1, 1)
        jtransform = rendering.Transform(rotation=-datt_rad[2],
                                         translation=np.array(
                                             [dpos[1], dpos[0]]))
        self.viewer.draw_polyline(
            ((24, 720), (24, 784), (40, 784), (40, 720), (24, 720)),
            filled=False)
        baz = np.clip(dpos[2] - 400, -400, 400)
        z_red_level = self.viewer.draw_polygon(
            ((24, 720), (24, 752 + baz / 400 * 32), (40, 752 + baz / 400 * 32),
             (40, 720)))
        z_red_level._color.vec4 = (0.83, 0.13, 0.18, 0.8)
        red_ac_img.add_attr(jtransform)
        self.viewer.onetime_geoms.append(red_ac_img)
        self.viewer.draw_polyline(dpos_hist[-50:, [-2, -3]],
                                  color=(0.9, 0.15, 0.2),
                                  linewidth=1.5)
        self.red_cone = self.make_cone(dpos, datt_rad[2])
        self.red_cone._color.vec4 = (.9, .15, .2, .3)
        transform2 = rendering.Transform(
            translation=(self.dpos[1], self.dpos[0]))  # Relative offset
        self.viewer.draw_circle(self.d_min, filled=False).add_attr(transform2)
        transform3 = rendering.Transform(
            translation=(dpos[1], dpos[0]))  # red dangerous circle
        self.viewer.draw_circle(self.d_max, filled=False).add_attr(transform3)

        # Blue aircraft
        apos, _, aatt_rad, apos_hist = self._blueAC.get_sta()
        self.viewer.draw_polyline(apos_hist[-50:, [-2, -3]],
                                  color=(0.00, 0.28, 0.73),
                                  linewidth=1.5)
        blue_ac_img = rendering.Image('envs/images/f16_blue.png', 48, 48)
        blue_ac_img._color.vec4 = (1, 1, 1, 1)
        jtransform = rendering.Transform(rotation=-aatt_rad[2],
                                         translation=np.array(
                                             [apos[1], apos[0]]))
        self.viewer.draw_polyline(
            ((56, 720), (56, 784), (72, 784), (72, 720), (56, 720)),
            filled=False)
        baz = np.clip(apos[2] - 400, -400, 400)
        z_blue_level = self.viewer.draw_polygon(
            ((56, 720), (56, 752 + baz / 400 * 32), (71, 752 + baz / 400 * 32),
             (71, 720)))
        z_blue_level._color.vec4 = (0.30, 0.65, 1.00, 0.8)
        blue_ac_img.add_attr(jtransform)
        self.viewer.onetime_geoms.append(blue_ac_img)
        self.blue_cone = self.make_cone(apos, aatt_rad[2])
        self.blue_cone._color.vec4 = (0.30, 0.65, 1.00, .3)

        # Health bars
        health_blue = self.viewer.draw_polygon(
            ((720, 80), (720, 112), (496 + self.red_health * 64, 112),
             (496 + self.red_health * 64, 80)))
        health_blue._color.vec4 = (0.30, 0.65, 1.00,
                                   0.8 - .25 * self.blue_health)
        health_blue = self.viewer.draw_polyline(
            ((720, 80), (720, 112), (496, 112), (496, 80), (720, 80)),
            color=(0.00, 0.00, 0.00),
            linewidth=3)
        health_red = self.viewer.draw_polygon(
            ((80, 80), (80, 112), (304 - self.red_health * 64, 112),
             (304 - self.red_health * 64, 80)))
        health_red._color.vec4 = (0.83, 0.13, 0.18,
                                  0.8 - .25 * self.red_health)
        health_red = self.viewer.draw_polyline(
            ((80, 80), (80, 112), (304, 112), (304, 80), (80, 80)),
            color=(0.00, 0.00, 0.00),
            linewidth=3)

        return self.viewer.render()

    def make_cone(self, position, head):
        foo1 = (position[1], position[0])
        foo2 = (position[1] +
                cos(-head + np.deg2rad(30) + np.pi / 2) * self.d_max,
                position[0] +
                sin(-head + np.deg2rad(30) + np.pi / 2) * self.d_max)
        foo3 = (position[1] +
                cos(-head - np.deg2rad(30) + np.pi / 2) * self.d_max,
                position[0] +
                sin(-head - np.deg2rad(30) + np.pi / 2) * self.d_max)
        return self.viewer.draw_polygon((foo1, foo2, foo3))

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def make_state(self, attacker, defender):
        apos, avel, aatt_rad, _ = attacker.get_sta()
        dpos, dvel, datt_rad, _ = defender.get_sta()
        self.distance_ = np.linalg.norm(apos - dpos)
        self.apos = apos
        self.dpos = dpos
        self.goal_pos = dpos
        self.goal_pos_defender = apos
        self.errPos = self.goal_pos - apos
        self.errPos_defender = self.goal_pos_defender - dpos
        posdiff = apos - dpos

        LOSxy = np.arctan2(self.errPos[1], self.errPos[0])
        LOSxy_defender = np.arctan2(self.errPos_defender[1],
                                    self.errPos_defender[0])
        posxy = np.linalg.norm(self.errPos[:2])
        posxy_defender = np.linalg.norm(self.errPos_defender[:2])
        LOSz = np.arctan2(self.errPos[2], posxy)
        LOSz_defender = np.arctan2(self.errPos_defender[2], posxy_defender)
        self.ATAxy_deg = np.rad2deg(self._pi_bound(-LOSxy + aatt_rad[2]))
        self.ATAz_deg = np.rad2deg(self._pi_bound(LOSz - aatt_rad[1]))
        self.AA_deg = np.rad2deg(self._pi_bound(datt_rad[2] - LOSxy))
        self.ATAxy_deg_defender = np.rad2deg(
            self._pi_bound(-LOSxy_defender + datt_rad[2]))
        self.ATAz_deg_defender = np.rad2deg(
            self._pi_bound(LOSz_defender - datt_rad[1]))
        self.AA_deg_defender = np.rad2deg(
            self._pi_bound(aatt_rad[2] - LOSxy_defender))
        self.qb = np.arccos(
            ((-posdiff[0]) * np.cos(aatt_rad[2]) * np.cos(aatt_rad[1]) -
             posdiff[1] * np.sin(aatt_rad[2]) * np.cos(aatt_rad[1]) +
             posdiff[2] * np.sin(aatt_rad[1])) / self.distance_)
        self.qb_deg = np.rad2deg(self.qb)
        self.qr = np.arccos(
            ((posdiff[0]) * np.cos(datt_rad[2]) * np.cos(datt_rad[1]) +
             posdiff[1] * np.sin(datt_rad[2]) * np.cos(datt_rad[1]) -
             posdiff[2] * np.sin(datt_rad[1])) / self.distance_)
        self.qr_deg = np.rad2deg(self.qr)
        self.vel_diff = avel - dvel

        return np.array([
            self.errPos[0], self.errPos[1], self.errPos[2],
            np.rad2deg(self._pi_bound(LOSxy)),
            np.rad2deg(self._pi_bound(LOSz)), self.ATAxy_deg, self.ATAz_deg,
            self.AA_deg, self.ATAxy_deg_defender, self.ATAz_deg_defender,
            self.AA_deg_defender,
            np.rad2deg(aatt_rad[0]),
            np.rad2deg(aatt_rad[1]),
            np.rad2deg(aatt_rad[2]), self.vel_diff, self.qr_deg, self.qb_deg
        ],
                        dtype=np.float32)

    def scalar_reward_terminal(self):
        terminal = False
        reward_sca = 0
        # qr - qb
        if abs(self.qr_deg) >= 90:
            reward_sca = -np.exp(abs(self.qr_deg) / 1800)
        elif abs(self.qr_deg) < 90:
            reward_sca = np.exp(-abs(self.qr_deg))
        if abs(self.qb_deg) >= 90:
            reward_sca -= np.exp(abs(self.qb_deg) / 1800)
        elif abs(self.qb_deg) < 90:
            reward_sca += np.exp(-abs(self.qb_deg))

        # Height
        if 0 < self.errPos[2] < 100:
            reward_sca += np.exp(abs(self.errPos[2]) / 1000)
        else:
            reward_sca -= np.exp(-abs(self.errPos[2]))

        # Velocity
        if 0 < self.vel_diff < 100:
            reward_sca += np.exp(abs(self.vel_diff) / 1000)
        else:
            reward_sca -= np.exp(-abs(self.vel_diff))

        info = " "
        reward_sca += np.exp(-self.distance_ / 1000)
        if self.distance_ <= self.d_min:  # collision
            reward_sca += -1500
            info = "collision"
            terminal = True
        elif abs(self.qr_deg) < 15 and abs(
                self.qb_deg) < 15 and 0 < self.errPos[2] < 200:
            reward_sca += 5
            if self.distance_ < 200:
                reward_sca += 1200
                terminal = True
                info = "win"
        elif abs(self.qr_deg) > 165 and abs(
                self.qb_deg) > 165 and 0 < self.errPos_defender[2] < 200:
            reward_sca -= 5
            if self.distance_ < 200:
                reward_sca -= 1200
                terminal = True
                info = "lost"
        return reward_sca, terminal, info

    def _pi_bound(self, u):
        u %= 2 * np.pi
        return u - 2 * np.pi if u > np.pi else u

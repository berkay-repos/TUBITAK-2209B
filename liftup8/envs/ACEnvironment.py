import numpy as np
import jsbsim
from math import sin, cos, tan
from config import Config


class ACEnvironment2D:
    def __init__(self,
                 position=np.array([0., 0., 0]),
                 att=np.array([0., 0., 0]),
                 vel_mps=0):
        self.reset()
        self._pos_m = position
        self._pos_history[0, :] = position
        self._att = att
        self._vel_mps = vel_mps
        self.dt = Config.action_time
        self.v_max = Config.vel_max_cmd
        self.v_min = Config.vel_min_cmd
        self.max_bank = Config.max_bank
        self.max_pitch = Config.max_pitch
        self.roll_rate = Config.roll_rate
        self.pitch_rate = Config.pitch_rate
        self.exec = jsbsim.FGFDMExec('./jsbsim/', None)
        self.exec.set_debug_level(0)
        self.exec.load_model("f16")
        self.exec.set_dt(Config.action_time)
        self.path = []

    def reset(self, USE_JSBSIM=False):
        if USE_JSBSIM:
            self.exec.set_property_value('ic/h-sl-ft', 8.42)
            self.exec.set_property_value('ic/terrain-elevation-ft', 8.42)
            self.exec.set_property_value('ic/h-agl-ft', 8.42)
            self.exec.set_property_value('ic/long-gc-deg', 1.37211666700005708)
            # geocentrique (angle depuis le centre de la terre)
            self.exec.set_property_value('ic/lat-gc-deg', 43.6189638890000424)
            # geodesique
            # self.exec.set_property_value('ic/lat-geod-deg', 43.6189638890000424)
            self.exec.set_property_value('ic/u-fps', 11.8147)
            self.exec.set_property_value('ic/v-fps', 0)
            self.exec.set_property_value('ic/w-fps', 0)
            self.exec.set_property_value('ic/p-rad_sec', 0)
            self.exec.set_property_value('ic/q-rad_sec', 0)
            self.exec.set_property_value('ic/r-rad_sec', 0)
            self.exec.set_property_value('ic/roc-fpm', 0)
            self.exec.set_property_value('ic/psi-true-deg', 0)
            self.exec.run_ic()
        else:
            self._pos_m = np.zeros((1, 3), dtype=float)
            self._att = np.zeros((1, 3), dtype=float)
            self._pos_history = np.zeros((1, 3), dtype=float)
            self._vel_mps = 0.

            # Should we use different dt for different commands or same?
            self._tau_flightpath_s = 0.05
            self._tau_vel_s = 0.1
            self._bank_tau = 0.2

    def get_sta(self):
        return self._pos_m.copy(), self._vel_mps, self._att.copy(
        ), self._pos_history

    def takeaction(self, Nx, Nz, mu):
        """Returns new state vector which is used for feature generation"""
        Nx = np.clip(Nx, self.v_min, self.v_max)
        vx = self._vel_mps * cos(self._att[1]) * cos(self._att[2])
        vy = self._vel_mps * cos(self._att[1]) * sin(self._att[2])
        vz = self._vel_mps * sin(self._att[1])
        self._att[0] += mu * np.deg2rad(self.roll_rate) * self._bank_tau
        self._att[0] = np.clip(self._att[0], -np.deg2rad(self.max_bank),
                               np.deg2rad(self.max_bank))
        self.vel_dot = (Nx - self._vel_mps) / self._tau_vel_s
        heading_dot_rad = 9.81 / self._vel_mps * tan(self._att[0]) * cos(
            self._att[1])
        self._att[1] += self.dt * np.deg2rad(self.pitch_rate) * Nz
        self._att[1] = np.clip(self._att[1], -np.deg2rad(self.max_pitch),
                               np.deg2rad(self.max_pitch))
        self._att[2] += self.dt * heading_dot_rad
        self._att[2] %= 2 * np.pi
        self._vel_mps += self.dt * self.vel_dot
        self._vel_mps = np.clip(self._vel_mps, self.v_min, self.v_max)
        self._pos_m += np.array([vx, vy, vz]) * self.dt
        self._pos_history = np.append(self._pos_history, [self._pos_m], axis=0)

        return np.array([
            self._pos_m.copy(), self._vel_mps,
            self._att.copy(), self._pos_history
        ],
                        dtype=object)

    def _pi_bound(self, u):  # Limiting to +/- 180 deg
        u %= 2 * np.pi
        return u - 2 * np.pi if u > np.pi else u

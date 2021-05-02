import numpy as np
from math import sin, cos, tan
from config import Config


class ACEnvironment2D:
    def __init__(self,
                 position=np.array([0., 0., 0]),
                 heading_deg=0.,
                 vel_mps=0):

        self.reset()
        self._pos_m = position
        self._pos_history[0, :] = position
        self._heading_rad = np.deg2rad(heading_deg)
        self._vel_mps = vel_mps
        self.dt = Config.action_time / 5
        self.v_max = Config.vel_max_cmd
        self.v_min = Config.vel_min_cmd
        self.max_bank = Config.max_bank
        self.roll_rate = Config.roll_rate
        self.pitch_rate = Config.pitch_rate

    def reset(self):
        self._pos_m = np.zeros((1, 3), dtype=float)
        self._pos_history = np.zeros((1, 3), dtype=float)
        self._vel_mps = 0.

        self._bank_rad = 0.
        self._flightpath_rad = 0.
        self._heading_rad = 0.

        # Should we use different dt for different commands or same?
        self._tau_flightpath_s = 0.05
        self._tau_vel_s = 0.1

    def get_sta(self):

        return np.array([
            self._pos_m.copy(), self._vel_mps,
            np.array([self._bank_rad, self._flightpath_rad, self._heading_rad],
                     dtype=object), self._pos_history
        ],
                        dtype=object)

    def takeaction(self, Nx, Nz, mu, bank_mode=False):
        """Returns new state vector which is used for feature generation"""
        if mu == 0:
            self._bank_rad = 0
        Nx = np.clip(Nx, self.v_min, self.v_max)
        vx = self._vel_mps * cos(self._flightpath_rad) * cos(self._heading_rad)
        vy = self._vel_mps * cos(self._flightpath_rad) * sin(self._heading_rad)
        vz = -self._vel_mps * sin(self._flightpath_rad)
        self._bank_rad += mu * np.deg2rad(self.roll_rate) * self.dt
        self._bank_rad = np.clip(self._bank_rad, -np.deg2rad(self.max_bank),
                                 np.deg2rad(self.max_bank))
        self.vel_dot = (Nx - self._vel_mps) / self._tau_vel_s
        heading_dot_rad = 9.81 / self._vel_mps * tan(self._bank_rad) * cos(
            self._flightpath_rad)
        self._flightpath_rad += self.dt * np.deg2rad(self.pitch_rate) * Nz
        self._flightpath_rad = np.clip(self._flightpath_rad, -10, 10)
        self._heading_rad += self.dt * heading_dot_rad
        self._vel_mps += self.dt * self.vel_dot
        self._vel_mps = np.clip(self._vel_mps, self.v_min, self.v_max)
        self._pos_m[0] += self.dt * vx
        self._pos_m[1] += self.dt * vy
        self._pos_m[2] += self.dt * vz
        self._pos_history = np.append(self._pos_history, [self._pos_m], axis=0)

        return np.array([
            self._pos_m.copy(), self._vel_mps,
            np.array([self._bank_rad, self._flightpath_rad, self._heading_rad],
                     dtype=object), self._pos_history
        ],
                        dtype=object)

    def _pi_bound(self, u):  # Limiting to +/- 180 deg
        u %= 2 * np.pi
        return u - 2 * np.pi if u > np.pi else u

    def _pi_bound_0_360(self, u):  # Limiting to 0-359 deg
        return np.mod(u, 2 * np.pi)

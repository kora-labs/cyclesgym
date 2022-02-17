import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


__all__ = ['InformedPolicy']


class InformedPolicy(object):
    """
    Policy with a fertilization window where the probability of actions depends on a difference between a learned saturation level and the current levels of N.

    The probability is given by
    pi(0|s) = pi1(0|s) + (1 - pi1(0|s)) * pi2(0|s)
    pi(a|s) = (1 - pi1(0|s)) * pi2(a|s)

    This means p1 indicates a fixed probability of selecting a=0 and p2 determines the remaining probability.
    pi1 changes depending on whether we are inside or outside the fertilization window.

    """
    def __init__(self, env, params):
        """
        The policy parametrization depends on the _parse_parameters function.

        Parameters
        ----------
        env: cyclesgym.env.CornEnv
            Environment to control
        params: list
            List of parameters
        """
        # Parameters for the fertilization window
        self.start_day = None  # start of window
        self.end_day = None # end of window
        self.a = None # steepness of switch from window open to closed
        self.max_val = None  # P(N=0) when window closed
        self.min_val = None  # P(N=0) when window open

        # p2 parameters
        self.saturation = None  # Max amount of N we want to put
        self.lengthscale = None  # Lengthscale of exponential decay of probability as a function of distance from saturation level

        self._parse_parameters(params)

        self.actions = env.maxN / (env.n_actions - 1) * np.arange(env.n_actions)

    def _parse_parameters(self, params):
        """
        Parse parameter vectors.

        This method is useful in case we want to change parametrization of the
        policy. Currently, the policy is parametrized by: start_day, end_day,
        a, max_val, min_val, saturation, lengthscale
        """
        # Better parametrization
        self.start_day = 1 + params[0] * 7
        delta = params[1] * 7
        self.end_day = np.clip(self.start_day + delta, a_min=0, a_max=366)
        self.a = 1/7
        self.max_val = params[2]
        self.min_val = params[3]

        # p2 parameters
        self.saturation = 20 * params[4]
        self.lengthscale = 10**params[5]

        # Naive parametrization
        # p1 parameters
        # self.start_day = params[0]
        # self.end_day = params[1]
        # self.a = 1/params[2]
        # self.max_val = params[3]
        # self.min_val = params[4]
        #
        # # p2 parameters
        # self.saturation = params[5]
        # self.lengthscale = params[6]

    def pi1(self, doy):
        """
        Probability of N=0 insisde and outside the fertilization window.

        Parameters
        ----------
        doy: int or np.array of ints
            Day of the year
        """
        doy = np.atleast_1d(doy)
        b1 = self.max_val + self.start_day * self.a
        b2 = self.max_val - self.end_day * self.a
        l1 = np.minimum(np.full_like(doy, self.max_val, dtype=float), -self.a * doy + b1)
        l2 = np.full_like(doy, self.min_val, dtype=float)
        l3 = np.minimum(np.full_like(doy, self.max_val, dtype=float), self.a * doy + b2)
        pi1 = np.maximum(np.maximum(l1, l2), l3)
        return pi1

    def pi2(self, deployed_N):
        """
        Probability of given amount of N depending on how much N has already been supplied.

        Parameters
        ----------
        doy: flaot or array of floats
            N already supplied
        """
        deployed_N = np.atleast_1d(deployed_N)
        delta = np.clip(self.saturation - deployed_N,
                        a_min=-self.actions[-1], a_max=self.actions[-1])

        Z = np.sum(
            np.exp(-(delta[:, None] - self.actions[None, :]) ** 2 / self.lengthscale),
            axis=1)
        pi2 = np.empty((self.actions.size, delta.size))
        for i, a in enumerate(self.actions):
            pi2[i, :] = np.exp(-(delta - a) ** 2 / self.lengthscale) / Z
        return pi2

    def action_probability(self, observation):
        """
        Determines probability of each based on the observation.

        Parameters
        ----------
        observation: np.ndarray
            First column: doy, second column: N
        """
        observation = np.atleast_2d(observation)
        doy = np.atleast_1d(observation[:, 0])
        deployed_N = np.atleast_1d(observation[:, 1])
        pi1 = self.pi1(doy)
        pi2 = self.pi2(deployed_N)
        pi = np.empty((doy.size, self.actions.size))
        for i, a in enumerate(self.actions):
            if a == 0:
                pi[:, i] = (1 - pi1) * pi2[i, :] + pi1
            else:
                pi[:, i] = (1 - pi1) * pi2[i, :]
        return pi

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Predic action probability based on observation with same interface as stable baselines policy.
        """
        probs = self.action_probability(observation)
        if deterministic:
            action = np.argmax(probs, axis=1)
        else:
            action = np.empty(probs.shape[0], dtype=int)
            for i, p in enumerate(probs):
                p/=sum(p)
                action[i] = np.random.choice(self.actions.size, size=1, p=p)

        return action, None

    def plot(self, doy, deployed_N):
        """
        Plot the give policy as a funciton of doy and deployed N.
        """
        doy = np.atleast_1d(doy)
        deployed_N = np.atleast_1d(deployed_N)
        if doy.size < 2 or deployed_N.size < 2:
            raise ValueError

        pi1 = self.pi1(doy)
        plt.figure()
        plt.plot(doy, pi1)
        plt.title('pi1')
        plt.xlabel('doy')

        plt.figure()
        pi2 = self.pi2(deployed_N)
        for a_distribution, a in zip(pi2, self.actions):
            plt.plot(deployed_N, a_distribution, label=f'{a}')
        plt.xlabel('Deployed N')
        plt.title('pi2')
        plt.legend()

        X, Y = np.meshgrid(doy, deployed_N)
        pi = np.empty((deployed_N.size, doy.size, self.actions.size))
        for i, a in enumerate(self.actions):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            if a == 0:
                pi[:, :, i] = (1 - pi1[None, :]) * pi2[i, :, None] + pi1
            else:
                pi[:, :, i] = (1 - pi1[None, :]) * pi2[i, :, None]
            ax.plot_surface(X, Y, pi[:, :, i], cmap=cm.coolwarm)
            plt.title(f'pi for action {a}')
            plt.xlabel('doy')
            plt.ylabel('deployed N')
        plt.show()


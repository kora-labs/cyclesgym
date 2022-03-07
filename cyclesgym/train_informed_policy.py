from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pygmo as pg
from stable_baselines3.common.evaluation import evaluate_policy

from cyclesgym.dummy_policies import OpenLoopPolicy
from cyclesgym.env import PartialObsCornEnv
from cyclesgym.informed_policy import InformedPolicy

from cyclesgym.paths import FIGURES_PATH


class informed_policy_problem(object):

    @staticmethod
    def _env_creator():
        """
        Create corn env with only doy and deployed N measurements.
        """
        mask = np.zeros(26, dtype=bool)
        mask[-2:] = True
        return PartialObsCornEnv('ContinuousCorn', delta=7, maxN=150, n_actions=11, mask=mask)

    def fitness(self, params):
        env = self._env_creator()
        model = InformedPolicy(env, params)
        mean_r, _ = evaluate_policy(model, env, n_eval_episodes=1,
                                    deterministic=True)
        return [-mean_r]

    def get_bounds(self):
        # Bounds for naive parametrization
        # return ([0, 150, 1, 0., 0, 0, 1],
        #         [150, 366, 100, 1, 1., 200, 1000])
        # Bounds for better parametrization
        return ([0, 0,  0.8, 0, 5, 0],
                [52, 52, 1, 0.2, 20, 3])


def evolve_pop_with_log(pop, algo, gen, maximize=True):
    """
    Evolve population for given problem and keep track of best f for each individual.

    Parameters
    ----------
    pop: pg.population
    algo: pg.algorithm
    gen: int
        Number of generations
    """
    n_individuals = len(pop.get_f())
    f_val = np.empty((gen + 1, n_individuals), dtype=float)
    f_val[0, :] = np.squeeze(pop.get_f())

    for i in range(gen):
        pop = algo.evolve(pop)
        f_val[i+1, :] = np.squeeze(pop.get_f())
    if maximize:
        f_val *= -1

    return pop, f_val


def create_action_sequence(doy, weight, maxN, n_actions, delta_t):
    doy = np.atleast_1d(doy)
    weight = np.atleast_1d(weight)
    assert len(doy) == len(weight)
    delta_a = maxN / (n_actions - 1)
    action_sequence = np.zeros(53, dtype=int)
    for d, w in zip(doy, weight):
        ind = np.floor(d / delta_t).astype(int)
        a = np.floor(w / delta_a).astype(int)
        action_sequence[ind] = a
    return action_sequence


if __name__ == '__main__':
    # Train GA
    p = informed_policy_problem()
    prob = pg.problem(p)
    algo = pg.algorithm(pg.sea(gen=1))
    pop = pg.population(prob, 5)
    pop, f_val = evolve_pop_with_log(pop, algo, gen=15, maximize=True)

    # Cycles expert policy
    env = p._env_creator()
    action_sequence = create_action_sequence(doy=110, weight=150,
                                             maxN=env.maxN,
                                             n_actions=env.n_actions,
                                             delta_t=env.delta)
    expert_policy_cycles = OpenLoopPolicy(action_sequence)
    cycles_expert_r, _ = evaluate_policy(expert_policy_cycles,
                                         env,
                                         n_eval_episodes=1,
                                         deterministic=True)
    # Agroscope expert policy 1
    action_sequence = create_action_sequence(doy=[110, 155], weight=[35, 120], # It should be 110 but we preder to overfertilize here
                                             maxN=env.maxN,
                                             n_actions=env.n_actions,
                                             delta_t=env.delta)
    expert_policy_agroscope = OpenLoopPolicy(action_sequence)
    agroscope_expert_r, _ = evaluate_policy(expert_policy_agroscope,
                                            env,
                                            n_eval_episodes=1,
                                            deterministic=True)


    # Plot
    plt.figure()
    plt.plot(f_val, linewidth=0.2, color='k', linestyle='--')
    plt.plot(np.full(f_val.shape[0],  cycles_expert_r),
             label='Cycles expert policy')
    plt.plot(np.full(f_val.shape[0], agroscope_expert_r),
             label='Agroscope expert policy')

    plt.plot(np.max(f_val, axis=1), color='k', label='Best GA solution')
    plt.legend(frameon=False)

    # Save plot
    plt.savefig(FIGURES_PATH.joinpath('GA_experts_comparison.pdf'), format='pdf', transparent=True)
    plt.ylim([1650, 2350])

    plt.show()



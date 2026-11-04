"""
ANFIS-GA (Hybrid) Model -- Section 6 of the paper
GA optimises ANFIS MF premise params (centers + sigmas).
GA parameters from paper Table 5:
  pop=100, gen=500, cx=0.80, mut=0.30, mut_sigma=0.02, tournament=8
"""

import numpy as np
import random
import sys
from deap import base, creator, tools, algorithms
from sklearn.metrics import r2_score
from src.anfis_model import ANFIS

# Avoid duplicate DEAP class registration errors on re-import
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


class ANFIS_GA:
    """
    ANFIS whose MF premise params are optimised by a Genetic Algorithm.

    Chromosome encoding (length = n_inputs * n_mfs * 2):
      [c_00, s_00, c_01, s_01, ..., c_(n-1)(m-1), s_(n-1)(m-1)]
    """

    def __init__(self, n_inputs=2, n_mfs=3, pop_size=100, n_generations=500,
                 cx_prob=0.80, mut_prob=0.30, mut_sigma=0.02,
                 tournament_size=8, epochs_anfis=50, random_state=42, **kwargs):
        self.n_inputs        = n_inputs
        self.n_mfs           = n_mfs
        self.pop_size        = pop_size
        self.n_generations   = n_generations
        self.cx_prob         = cx_prob
        self.mut_prob        = mut_prob
        self.mut_sigma       = mut_sigma
        self.tournament_size = tournament_size
        self.epochs_anfis    = epochs_anfis
        self.rs              = random_state
        self.chromosome_len  = n_inputs * n_mfs * 2
        self.best_anfis      = None
        self.ga_logbook      = None

    def _decode(self, chromosome, model):
        """Apply chromosome to model MF centers and sigmas."""
        idx = 0
        for i in range(self.n_inputs):
            for k in range(self.n_mfs):
                model.mfs[i][k].center = chromosome[idx]
                model.mfs[i][k].sigma  = max(1e-4, abs(chromosome[idx + 1]))
                idx += 2

    def _fitness(self, chromosome, X_tr, y_tr, X_val, y_val):
        """
        Fitness = (f1 + f2) / 2  (paper Section 2.4)
        f1 = training RMSE, f2 = validation RMSE
        """
        model = ANFIS(n_inputs=self.n_inputs, n_mfs=self.n_mfs,
                      epochs=self.epochs_anfis,
                      random_state=self.rs)
        model._init_mfs(X_tr)
        model.consequents = np.zeros((model.n_rules, model.n_inputs + 1))
        self._decode(chromosome, model)

        # Fit only consequents (premise fixed by chromosome)
        model._fit_consequents(X_tr, y_tr)

        y_hat_tr  = model.predict(X_tr)
        y_hat_val = model.predict(X_val)

        f1 = np.sqrt(np.mean((y_hat_tr  - y_tr)  ** 2))
        f2 = np.sqrt(np.mean((y_hat_val - y_val) ** 2))
        return ((f1 + f2) / 2,)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        random.seed(self.rs)
        np.random.seed(self.rs)

        # Internal 80/20 split for validation if none provided
        if X_val is None:
            sp    = int(0.8 * len(X_train))
            X_val, y_val = X_train[sp:], y_train[sp:]
            X_tr,  y_tr  = X_train[:sp], y_train[:sp]
        else:
            X_tr, y_tr = X_train, y_train

        print(f"\n{'='*60}")
        print(f"  ANFIS-GA | pop={self.pop_size}  gen={self.n_generations}")
        print(f"  cx={self.cx_prob}  mut={self.mut_prob}  sigma={self.mut_sigma}  tourn={self.tournament_size}")
        print(f"{'='*60}")
        sys.stdout.flush()

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.gauss, 0.5, 0.2)
        toolbox.register("individual", tools.initRepeat,
                         creator.Individual, toolbox.attr_float, n=self.chromosome_len)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._fitness,
                         X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val)
        toolbox.register("mate",   tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=self.mut_sigma, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        hof = tools.HallOfFame(1)

        population = toolbox.population(n=self.pop_size)

        # Manual EA loop so we can print & flush every 50 generations
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)
        hof.update(population)

        log = []
        for gen in range(1, self.n_generations + 1):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values

            # Mutation
            for mut in offspring:
                if random.random() < self.mut_prob:
                    toolbox.mutate(mut)
                    del mut.fitness.values

            # Evaluate new individuals
            invalids = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalids:
                ind.fitness.values = toolbox.evaluate(ind)

            # Strict Elitism: guarantee best individual is preserved in next generation
            if len(hof) > 0 and hof[0].fitness.valid:
                offspring[0] = toolbox.clone(hof[0])

            population[:] = offspring
            hof.update(population)

            fits = [ind.fitness.values[0] for ind in population]
            record = {"gen": gen, "min": min(fits), "avg": float(np.mean(fits))}
            log.append(record)

            if gen % 50 == 0 or gen == 1:
                print(f"  Gen [{gen:>3}/{self.n_generations}]  "
                      f"Best RMSE: {record['min']:.5f}  "
                      f"Avg RMSE:  {record['avg']:.5f}")
                sys.stdout.flush()

        self.ga_logbook = log

        # Refit best ANFIS on full training data
        best_model = ANFIS(n_inputs=self.n_inputs, n_mfs=self.n_mfs,
                           epochs=self.epochs_anfis,
                           random_state=self.rs)
        best_model._init_mfs(X_train)
        best_model.consequents = np.zeros((best_model.n_rules, best_model.n_inputs + 1))
        self._decode(hof[0], best_model)

        best_model._fit_consequents(X_train, y_train)

        self.best_anfis = best_model
        print(f"{'='*60}")
        print(f"  GA done. Best fitness: {hof[0].fitness.values[0]:.5f}")
        print(f"{'='*60}")
        sys.stdout.flush()
        return self

    def predict(self, X):
        return self.best_anfis.predict(X)

    def score(self, X, y):
        return r2_score(y, self.predict(X))
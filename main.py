#!/usr/bin/env python
import random
import numpy as np
import pandas as pd
import multiprocessing
from copy import deepcopy
from datetime import datetime
import pickle


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

import seaborn
seaborn.set(style='whitegrid')

from scoop import futures

from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import base, creator, tools

from sklearn.model_selection import (train_test_split,
                                     cross_val_score, 
                                     cross_val_predict, 
                                     KFold,
                                     ShuffleSplit,  
                                     StratifiedKFold,
                                     StratifiedShuffleSplit)

from sklearn.metrics import (accuracy_score,
                             confusion_matrix)

# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from funcs import *
from plots import *


##################################################
#
# Constants
#
##################################################
RANDOMSTATE = 42

#
# ML evaluation attributes
#

# Number of splits for accuracy evaluation
NSPLITS = 5
# Size of test 
TESTSIZE = .3

#
# GA attributes
#

# Population size need to be multiple 4
POPSIZE = 500
# Maximum number of generations
NGEN = 10
# Crossover probability
CXPB = 0.9
# Mutation probability
MUTPB = 0.2

################################################################
#
# Creating our genetic algorithim
#
################################################################
# The optimization parameter
# We want to maximize the accuracy (first parameter) and minimize the number
# of parameters used.
creator.create('Fitness', base.Fitness, weights=(1.0, 1.0))

# Ou individual is related with the fitness
creator.create('Individual', list, fitness=creator.Fitness)


if __name__ == '__main__':
    # Local functions
    random.seed(RANDOMSTATE)
    sonar_data, sonar_target, sonar_target_classes, sonar_names = load_sonar_dataset('data/sonar/sonar.all-data')
    n_samples, n_features = sonar_data.shape

    # Initialize the genetic toolkit
    tb = base.Toolbox()

    # A feature is a binary value
    tb.register('attr_feature', random.randint, 0, 1)

    # A individual is a colection of bool values related to each feature
    tb.register('individual', tools.initRepeat, creator.Individual, tb.attr_feature, n=n_features)

    # Defines the population
    tb.register('population', tools.initRepeat, list, tb.individual)

    tb.register('mate', tools.cxTwoPoint)
    tb.register('mutate', tools.mutFlipBit, indpb=0.1)
    #tb.register('map', futures.map)
    #tb.register('map', pool.map_async)

    tb.register('select', tools.selNSGA2)

    tb.pop_size = POPSIZE
    tb.max_gen = NGEN
    tb.mut_prob = MUTPB

    #
    # Segment Training / Test data
    #
    train_X, test_X, train_y, test_y = train_test_split(sonar_data, 
                                                        sonar_target, 
                                                        stratify=sonar_target, 
                                                        test_size=TESTSIZE,
                                                        shuffle=True, 
                                                        random_state=RANDOMSTATE)



    # total_evals = 500
    # pop_sizes = np.array([10, 50, 100])
    # gen_sizes = total_evals/pop_sizes

    stats_values = tools.Statistics(fitness_values)
    stats_values.register('min', np.min, axis=0)
    stats_values.register('max', np.max, axis=0)
    stats_values.register('std', np.std, axis=0)
    stats_values.register('avg', np.mean, axis=0)

    stats_features = tools.Statistics(features)
    stats_features.register('hvol', evaluate_hypervolume)
    stats_features.register('sum', np.sum, axis=1)
    stats_features.register('hist', np.sum, axis=0)
    stats_features.register('feat', features)

    stats_runtime = tools.Statistics(now_timestamp)
    stats_runtime.register('start', now_timestamp)

    mstats = tools.MultiStatistics(fitness=stats_values, features=stats_features, runtime=stats_runtime)

    logbook = tools.Logbook()
    logbook.header = 'gen', 'evals', 'fitness', 'features', 'runtime'
    logbook.chapters['fitness'].header =  'min', 'max', 'std', 'avg'
    logbook.chapters['features'].header = 'hvol', #, 'sum' 'hist', 'feat'
    logbook.chapters['runtime'].header = 'start', 

    classifier = GaussianNB()
    # classifier = MLPClassifier(solver='lbfgs', activation='relu', alpha=1e-5, max_iter=1000, hidden_layer_sizes=(100,60), random_state=RANDOMSTATE)
    cv = StratifiedShuffleSplit(n_splits=NSPLITS, test_size=TESTSIZE, random_state=RANDOMSTATE)
    #cv = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=RANDOMSTATE)

    tb.register('evaluate', evaluate_ml_classifier, data=test_X, target=test_y, classifier=classifier, cross_val=cv)
    tb.decorate('evaluate', tools.DeltaPenalty(feasible, (0, -60)))
    # Initial population
    pop = tb.population(n=POPSIZE)
    invalid = evaluate_population_fitness(tb, pop)
    pop = tb.select(pop, len(pop))


    np.set_printoptions(precision=2)
    record = mstats.compile(pop)
    logbook.record(gen=0, evals=invalid, **record)
    print(logbook.stream)


    # Iterative mode on
    plt.ion()
    fig, ax = plt.subplots(dpi=150)
    ax.set_xlim([0,1])
    ax.set_ylim([-60,0])
    plot_population_fitness(pop, ax, color='black', alpha=.5, sizes=[2], clear=True)
    ax.set_xlim([0,1])
    ax.set_ylim([-60,0])
    plt.pause(0.05)


    for gen in range(1, NGEN):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [tb.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                tb.mate(ind1, ind2)

            tb.mutate(ind1)
            tb.mutate(ind2)

            del ind1.fitness.values, ind2.fitness.values

        invalid = evaluate_population_fitness(tb, offspring)
        pop = tb.select(pop + offspring, POPSIZE)
        record = mstats.compile(pop)
        logbook.record(gen=gen, evals=invalid, **record)
        ax.set_xlim([0,1])
        ax.set_ylim([-60,0])
        plot_population_fitness(pop, ax, color='black', alpha=.5, sizes=[2], clear=True)
        ax.set_xlim([0,1])
        ax.set_ylim([-60,0])
        plt.pause(0.05)
        print(logbook.stream)


    # Picling to disk
    now_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    with open(f'pickles/{now_timestamp}_logbook.bin', 'wb') as fd:
        pickle.dump(logbook, fd)

    # with open(f'pickles/{now_timestamp}_logbook.bin', 'rb') as fd:
    #     readback = pickle.load(fd)

    # OTHER HYPER PARAMETERS AND VALIDATIONS
    # # Picking the features (the individual)
    feats = logbook.chapters['features'].select('feat')
    histo = logbook.chapters['features'].select('hist')
    hvol = logbook.chapters['features'].select('hvol')

    # # Plot population
    # seaborn.set(style='whitegrid')
    # fig, ax = plt.subplots(dpi=150)
    # plot_population_fitness(feats[0], ax, color='blue', alpha=.5, sizes=[4], label=f'Gen 1')
    # plot_population_fitness(feats[NGEN//2], ax, color='purple', alpha=.5, sizes=[4], label=f'Gen {NGEN//2}')
    # plot_population_fitness(feats[-1], ax, color='red', alpha=.5, sizes=[4], label=f'Gen {NGEN}')
    # ax.set_xlim([0,1])
    # ax.set_ylim([-60,0])
    # ax.set_title('População')
    # fig.legend(loc='lower left')
    # plt.tight_layout()
    # fig.savefig('imgs/last_pop.png', dpi=150)
    #  
    #  
    # # Plot histograms
    # fig, ax = plt.subplots(dpi=150)
    # plot_feature_histogram(histo[0], ax)
    # plt.tight_layout()
    # fig.savefig('imgs/feature_histogram_0_first.png', dpi=150)

    # fig, ax = plt.subplots(dpi=150)
    # plot_feature_histogram(histo[NGEN//2], ax)
    # plt.tight_layout()
    # fig.savefig('imgs/feature_histogram_1_half.png', dpi=150)

    # fig, ax = plt.subplots(dpi=150)
    # plot_feature_histogram(histo[-1], ax)
    # plt.tight_layout()
    # fig.savefig('imgs/feature_histogram_2_last.png', dpi=150)


    # # Plot hypervolume evolution
    # fig, ax = plt.subplots(dpi=150)
    # ax.plot(hvol)
    # ax.set_title('Evolução do Hipervolume')
    # ax.set_xlabel('Geração')
    # ax.set_ylabel('Volume')
    # fig.tight_layout()
    # fig.savefig('imgs/hypervol_evolution.png', dpi=150)

    #  
    # # Plot feature selection evolution
    # seaborn.set(style='white')
    # fig, ax = plt.subplots(dpi=150)
    # plot_feature_selection(np.array(histo)/POPSIZE, ax)
    # plt.tight_layout()
    # fig.savefig('imgs/feature_evolution.png', dpi=150)


    # Plot pareto front
    #seaborn.set(style='whitegrid')
    #fig, ax = plt.subplots(dpi=150)
    #plot_pareto_fronts(feats[7], ax, nfronts=3)
    #ax.set_xlim([0,1])
    #ax.set_ylim([-60,0])
    #ax.set_xlabel('Acurácia')
    #ax.set_ylabel('# Atributos')
    #ax.set_title('Frente de Pareto')
    #ax.legend(loc='lower left')
    #plt.tight_layout()
    #fig.savefig('imgs/pareto_front.png', dpi=150)



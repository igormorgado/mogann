#!/usr/bin/env python
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.model_selection import cross_val_score 
from deap import tools
from deap.benchmarks.tools import hypervolume
from datetime import datetime



def load_sonar_dataset(filepath):
    """Loads sonar data and returns a tuple with
        sonar_data
        sonar_target
        sonar_target_classes
        sonar_column_names
    """
    sonar_columns = [ f'b{x:02d}' for x in np.arange(60) ]
    sonar_columns.append('target')

    # Read dataset
    sonar_df = pd.read_csv(filepath, names=sonar_columns)

    # Create a numpy representation
    sonar_data = sonar_df.values[:, :-1].astype('float')
    sonar_target = sonar_df.values[:, -1].astype(str)
    sonar_target_classes = np.array(['Mine', 'Rock'])

    return sonar_data, sonar_target, sonar_target_classes, sonar_columns

def feasible(individual):
    return np.sum(individual) > 0

def evaluate_ml_classifier(individual, data, target, classifier, cross_val):
    """Evaluates the fitness of a individual (accuracy) given input parameters
        data
        target
        classifier
        validation shuffling
    Returns the tuple with individual values"""
    subdata = data[:, np.flatnonzero(individual)]
    cv_scores = cross_val_score(classifier, subdata, target, cv=cross_val, verbose=0)
    return cv_scores.mean(),-sum(individual)

def now_timestamp(individual):
    return datetime.now().strftime('%Y%m%d%H%M%S')


def fitness_values(individual):
    """Return the fitness values of a given individual"""
    return individual.fitness.values

def features(individual):
    """Return the fitness values of a given individual"""
    return individual

def evaluate_population_fitness(toolbox, population):
    """Evaluates population fitness updating each individual fitness.
    Returns the number of invalid population updated"""
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    return len(invalid_ind)

def pareto_dominance(ind1, ind2):
    """Returns a bool value regarding ind1 dominates ind2"""
    return tools.emo.isDominated(ind1.fitness.values, ind2.fitness.values)

def evaluate_hypervolume(data):
    return hypervolume(data, [1, 60])


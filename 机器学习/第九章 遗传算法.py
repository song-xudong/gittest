# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 17:54:51 2022

作者：李一邨
人工智能算法案例大全：基于Python
浙大城市学院、杭州伊园科技有限公司
浙江大学 博士
中国民盟盟员
Email:liyicun_yykj@163.com


"""

#%%


import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt

#%%


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

#%%


class Fitness:
    def __init__(self, route):
        self.route = route 
        self.distance = 0
        self.fitness= 0.0
    
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                
                if i + 1 < len(self.route): 
                    toCity = self.route[i + 1] 
                else: 
                    toCity = self.route[0] 
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

#%%


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

#%%

def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


# Note: we only have to use these functions to create the initial population. Subsequent generations will be produced through breeding and mutation.

#%%

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

#%%




def selection(popRanked, eliteSize):
    selectionResults = []
    
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    
    
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

#%%


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

#%%


def crossover(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    
    
    for i in range(startGene, endGene):
        childP1.append(parent1[i])
    
    
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

#%%


def crossoverPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))
    
    
    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    
    for i in range(0, length):
        child = crossover(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

#%%


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

#%%


def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

#%%


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = crossoverPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


#%%

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        print("Best distance so far: " + str(1 / rankRoutes(pop)[0][1]))
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

#%%


cityList = []

for i in range(0,20):
    cityList.append(City(x=int(random.random() * 200), 
                         y=int(random.random() * 200)))
    
#%%


geneticAlgorithm(population=cityList, 
                 popSize=50, 
                 eliteSize=5, 
                 mutationRate=0.01, 
                 generations=500)

#%%


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
    
#%%


geneticAlgorithmPlot(population=cityList, 
                 popSize=50, 
                 eliteSize=5, 
                 mutationRate=0.01, 
                 generations=100)

#%%



#%%


import random
import numpy as np
import matplotlib.pyplot as plt

#%%


from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
 
SEED = 2022
random.seed(SEED)
np.random.seed(SEED)

dataset = load_boston()
X, y = dataset.data, dataset.target
features = dataset.feature_names
print(X)
print(y)
print(features)

#%%


est = LinearRegression()
score = -1.0 * cross_val_score(est, X, y, cv=5, scoring="neg_mean_squared_error")
print("CV MSE before feature selection: {:.2f}".format(np.mean(score)))



#%%


class GeneticSelector():
    def __init__(self, estimator, n_gen, size, n_best, n_rand, 
                 n_children, mutation_rate):
        # Estimator 
        self.estimator = estimator
        # Number of generations
        self.n_gen = n_gen
        # Number of chromosomes in population
        self.size = size
        # Number of best chromosomes to select (Elitism)
        self.n_best = n_best
        # Number of random chromosomes to select
        self.n_rand = n_rand
        # Number of children created during crossover
        self.n_children = n_children
        # Probablity of chromosome mutation
        self.mutation_rate = mutation_rate
        
        if int((self.n_best + self.n_rand) / 2) * self.n_children != self.size:
            raise ValueError("The population size is not stable.")
        


#%%


def initilize(self):
    population = []
    
    for i in range(self.size):
        chromosome = np.ones(self.n_features, dtype=np.bool)
        mask = np.random.rand(len(chromosome)) < 0.3
        chromosome[mask] = False
        population.append(chromosome)
    
    return population

GeneticSelector.initilize = initilize 



    
    
#%%


def fitness(self, population):
    X, y = self.dataset
    scores = []
    for chromosome in population:
        #Score is the MSE
        score = -1.0 * np.mean(cross_val_score(self.estimator, 
                                               X[:,chromosome],
                                               y, 
                                               cv=5, 
                                               scoring="neg_mean_squared_error"))
        scores.append(score)
        
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)
    
    
    return list(scores[inds]), list(population[inds,:])

GeneticSelector.fitness = fitness



#%%



def select(self, population_sorted):
    population_next = []
    
    for i in range(self.n_best):
        population_next.append(population_sorted[i])
        
    for i in range(self.n_rand):
        population_next.append(random.choice(population_sorted))
        
    random.shuffle(population_next)
    
    return population_next

GeneticSelector.select = select


#%%


def crossover(self, population):
    population_next = []
    
    for i in range(int(len(population)/2)):
         for j in range(self.n_children):
            chromosome1, chromosome2 = population[i], population[len(population)-1-i]
            child = chromosome1
            mask = np.random.rand(len(child)) > 0.5
            child[mask] = chromosome2[mask]
            population_next.append(child)
            
    return population_next

GeneticSelector.crossover = crossover


#%%


def mutate(self, population):
    population_next = []
    for i in range(len(population)):
        chromosome = population[i]
        if random.random() < self.mutation_rate:
            mask = np.random.rand(len(chromosome)) < 0.05
            chromosome[mask] = False
        population_next.append(chromosome)
        
    return population_next

GeneticSelector.mutate = mutate


#%%


def generate(self, population):
    
    scores_sorted, population_sorted = self.fitness(population)
    population = self.select(population_sorted)
    population = self.crossover(population)
    population = self.mutate(population)
    
    self.chromosomes_best.append(population_sorted[0])
    self.scores_best.append(scores_sorted[0])
    self.scores_avg.append(np.mean(scores_sorted))

    return population

GeneticSelector.generate = generate


#%%


def fit(self, X, y):

    self.chromosomes_best = []
    self.scores_best, self.scores_avg  = [], []

    self.dataset = X, y
    self.n_features = X.shape[1]

    population = self.initilize()
    for i in range(self.n_gen):
        population = self.generate(population)

    return self 

@property
def support_(self):
    return self.chromosomes_best[-1]

def plot_scores(self):
    plt.plot(self.scores_best, label='Best')
    plt.plot(self.scores_avg, label='Average')
    plt.legend()
    plt.ylabel('Scores')
    plt.xlabel('Generation')
    plt.show()
    
GeneticSelector.fit = fit
GeneticSelector.support_ = support_
GeneticSelector.plot_scores = plot_scores

#%%


sel = GeneticSelector(estimator=LinearRegression(), 
                      n_gen=100, size=200, n_best=40, n_rand=40, 
                      n_children=5, mutation_rate=0.05)
sel.fit(X, y)
sel.plot_scores()
score = -1.0 * cross_val_score(est, X[:,sel.support_], y, cv=5, scoring="neg_mean_squared_error")
print("CV MSE after feature selection: {:.2f}".format(np.mean(score)))

#%%



print('Select features are: ', features[sel.chromosomes_best[4]])
print('Score: ', round(sel.scores_best[4],3))

















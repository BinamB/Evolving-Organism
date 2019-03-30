from __future__ import division
from collections import defaultdict

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines

from plotting import *

import numpy as np
import operator

from math import atan2
from math import cos
from math import degrees
from math import floor
from math import radians
from math import sin
from math import sqrt
from random import randint
from random import random
from random import sample
from random import uniform
#--- CONSTANTS ----------------------------------------------------------------+

settings = {}

# EVOLUTION SETTINGS 
settings['pop_size'] = 30      # number of organisms
settings['food_num'] = 30      # number of food particles
settings['superfood_num'] = 0     # number of food particles
settings['poison_num'] = 30    # number of poison particles
settings['gens'] = 20        # number of generations
settings['elitism'] = 0.20     # elitism (selection bias)
settings['mutate'] = 0.10      # mutation rate

# SIMULATION SETTINGS
settings['gen_time'] = 50      # generation length         (seconds)
settings['dt'] = 0.04          # simulation time step      (dt)
settings['dr_max'] = 720       # max rotational speed      (degrees per second)
settings['v_max'] = 0.5        # max velocity              (units per second)
settings['dv_max'] =  0.25     # max acceleration (+/-)    (units per second^2)

settings['x_min'] = -2.0       # arena western border
settings['x_max'] =  2.0       # arena eastern border
settings['y_min'] = -2.0       # arena southern border
settings['y_max'] =  2.0       # arena northern border

settings['plot'] = True        # plot final generation?

# ORGANISM NEURAL NET SETTINGS
settings['inodes'] = 1         # number of input nodes
settings['hnodes'] = 5         # number of hidden nodes
settings['onodes'] = 2         # number of output nodes

#--- FUNCTIONS ----------------------------------------------------------------+

def dist(x1,y1,x2,y2):
    return sqrt((x2-x1)**2 + (y2-y1)**2)


def calc_heading(org, food):
    d_x = food.x - org.x
    d_y = food.y - org.y
    theta_d = degrees(atan2(d_y, d_x)) - org.r
    if abs(theta_d) > 180: theta_d += 360
    return theta_d / 180

def calc_heading_superfood(org, superfood):
    d_x = superfood.x - org.x
    d_y = superfood.y - org.y
    theta_d = degrees(atan2(d_y, d_x)) - org.r
    if abs(theta_d) > 180: theta_d += 360
    return theta_d / 180


def calc_heading_poison(org, poison):
    d_x = poison.x - org.x
    d_y = poison.y - org.y
    theta_d = degrees(atan2(d_y, d_x)) - org.r
    if abs(theta_d) > 180: theta_d += 360
    return theta_d / 180


def plot_frame(settings, organisms, foods, superfoods, poisons, gen, time):
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 5)
                        
    plt.xlim([settings['x_min'] + settings['x_min'] * 0.25, settings['x_max'] + settings['x_max'] * 0.25])
    plt.ylim([settings['y_min'] + settings['y_min'] * 0.25, settings['y_max'] + settings['y_max'] * 0.25])

    # PLOT ORGANISMS
    for organism in organisms:
        plot_organism(organism.x, organism.y, organism.r, ax)

    # PLOT FOOD PARTICLES
    for food in foods:
        plot_food(food.x, food.y, ax)

    # PLOT SUPERFOOD PARTICLES
    for superfood in superfoods:
        plot_superfood(superfood.x, superfood.y, ax)    

    #PLOT POISON PARTICLES
    for poison in poisons:
        plot_poison(poison.x, poison.y, ax)

    # MISC PLOT SETTINGS
    ax.set_aspect('equal')
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])

    plt.figtext(0.025, 0.95,r'GENERATION: '+str(gen))
    plt.figtext(0.025, 0.90,r'T_STEP: '+str(time))

    plt.savefig(str(gen)+'-'+str(time)+'.png', dpi=100)
##    plt.show()


def evolve(settings, organisms_old, gen):

    elitism_num = int(floor(settings['elitism'] * settings['pop_size']))
    new_orgs = settings['pop_size'] - elitism_num

    #--- GET STATS FROM CURRENT GENERATION ----------------+
    stats = defaultdict(int)
    for org in organisms_old:
        if org.fitness > stats['BEST'] or stats['BEST'] == 0:
            stats['BEST'] = org.fitness

        if org.fitness < stats['WORST'] or stats['WORST'] == 0:
            stats['WORST'] = org.fitness
            
        stats['SUM'] += org.fitness
        stats['COUNT'] += 1

    stats['AVG'] = stats['SUM'] / stats['COUNT']
    
    
    #--- ELITISM (KEEP BEST PERFORMING ORGANISMS) ---------+
    orgs_sorted = sorted(organisms_old, key=operator.attrgetter('fitness'), reverse=True)
    organisms_new = []
    for i in range(0, elitism_num):
        organisms_new.append(organism(settings, wih=orgs_sorted[i].wih, who=orgs_sorted[i].who, name=orgs_sorted[i].name))

    
    #--- GENERATE NEW ORGANISMS ---------------------------+
    for w in range(0, new_orgs):

        # SELECTION (TRUNCATION SELECTION)
        canidates = range(0, elitism_num)
        random_index = sample(canidates, 2)
        org_1 = orgs_sorted[random_index[0]]
        org_2 = orgs_sorted[random_index[1]]

        # CROSSOVER
        crossover_weight = random()
        wih_new = (crossover_weight * org_1.wih) + ((1 - crossover_weight) * org_2.wih)
        who_new = (crossover_weight * org_1.who) + ((1 - crossover_weight) * org_2.who)
        
        # MUTATION
        mutate = random()
        if mutate <= settings['mutate']:

            # PICK WHICH WEIGHT MATRIX TO MUTATE
            mat_pick = randint(0,1)     

            # MUTATE: WIH WEIGHTS
            if mat_pick == 0:
                index_row = randint(0,settings['hnodes']-1)
                wih_new[index_row] = wih_new[index_row] * uniform(0.9, 1.1)
                if wih_new[index_row] >  1: wih_new[index_row] = 1
                if wih_new[index_row] < -1: wih_new[index_row] = -1
                
            # MUTATE: WHO WEIGHTS
            if mat_pick == 1:
                index_row = randint(0,settings['onodes']-1)
                index_col = randint(0,settings['hnodes']-1)
                who_new[index_row][index_col] = who_new[index_row][index_col] * uniform(0.9, 1.1)
                if who_new[index_row][index_col] >  1: who_new[index_row][index_col] = 1
                if who_new[index_row][index_col] < -1: who_new[index_row][index_col] = -1
                    
        organisms_new.append(organism(settings, wih=wih_new, who=who_new, name='gen['+str(gen)+']-org['+str(w)+']'))
                
    return organisms_new, stats


def simulate(settings, organisms, foods, superfoods, poisons, gen):

    total_time_steps = int(settings['gen_time'] / settings['dt'])
    
    #--- CYCLE THROUGH EACH TIME STEP ---------------------+
    for t_step in range(0, total_time_steps, 1):

        # PLOT SIMULATION FRAME
        if settings['plot']==True and gen==settings['gens']-1:
            plot_frame(settings, organisms, foods, superfoods, poisons, gen, t_step)
        
        # UPDATE FITNESS FUNCTION
        for food in foods:
            for org in organisms:
                food_org_dist = dist(org.x, org.y, food.x, food.y)

                # UPDATE FITNESS FUNCTION
                if food_org_dist <= 0.075:
                    org.fitness += food.energy
                    food.respawn(settings)
                
                # RESET DISTANCE AND HEADING TO NEAREST FOOD SOURCE
                org.d_food = 100
                org.r_food = 0

        # UPDATE FITNESS FUNCTION
        for superfood in superfoods:
            for org in organisms:
                superfood_org_dist = dist(org.x, org.y, superfood.x, superfood.y)

                # UPDATE FITNESS FUNCTION
                if superfood_org_dist <= 0.075:
                    org.fitness = org.fitness + (superfood.energy)
                    superfood.respawn(settings)
                
                # RESET DISTANCE AND HEADING TO NEAREST FOOD SOURCE
                org.d_superfood = 100
                org.r_superfood = 0
                
                    
        # UPDATE FITNESS FUNCTION
        for poison in poisons:
            for org in organisms:
                poison_org_dist = dist(org.x, org.y, poison.x, poison.y)

                # UPDATE FITNESS FUNCTION
                if poison_org_dist <= 0.075:
                    org.fitness += poison.energy
                    poison.respawn(settings)

             
                # RESET DISTANCE AND HEADING TO NEAREST POISON
                org.d_poison = 100
                org.r_poison = 0


# CALCULATE HEADING TO NEAREST SUPERFOOD SOURCE
        for superfood in superfoods:
            for org in organisms:
                
                # CALCULATE DISTANCE TO SELECTED FOOD PARTICLE
                superfood_org_dist = dist(org.x, org.y, superfood.x, superfood.y)
                food_org_dist = dist(org.x, org.y, food.x, food.y)

                # DETERMINE IF THIS IS THE CLOSEST FOOD PARTICLE
                if (superfood_org_dist < org.d_superfood):
                    org.d_superfood = superfood_org_dist
                    org.r_superfood = calc_heading_superfood(org, superfood)


        # CALCULATE HEADING TO NEAREST FOOD SOURCE
        for food in foods:
            for org in organisms:
                
                # CALCULATE DISTANCE TO SELECTED FOOD PARTICLE
                food_org_dist = dist(org.x, org.y, food.x, food.y)

                # DETERMINE IF THIS IS THE CLOSEST FOOD PARTICLE
                if food_org_dist < org.d_food:
                    org.d_food = food_org_dist
                    org.r_food = calc_heading(org, food)
         
         # CALCULATE HEADING TO NEAREST POISON
        for poison in poisons:
            for org in organisms:
                
                # CALCULATE DISTANCE TO SELECTED POISON PARTICLE
                poison_org_dist = dist(org.x, org.y, poison.x, poison.y)

                # DETERMINE IF THIS IS THE CLOSEST POISON PARTICLE
                if poison_org_dist < org.d_poison:
                    org.d_poison = poison_org_dist
                    org.r_poison = calc_heading_poison(org, poison)

        # GET ORGANISM RESPONSE
        for org in organisms:
            org.think()

        # UPDATE ORGANISMS POSITION AND VELOCITY
        for org in organisms:
            org.update_r(settings)
            org.update_vel(settings)
            org.update_pos(settings)

    return organisms


#--- CLASSES ------------------------------------------------------------------+


class food():
    def __init__(self, settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = 1


    def respawn(self,settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = 1
        
class superfood():
    def __init__(self, settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = 5


    def respawn(self,settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = 5



class poison():
    def __init__(self, settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = -1


    def respawn(self,settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = -1



class organism():
    def __init__(self, settings, wih=None, who=None, name=None):
        
        self.x = uniform(settings['x_min'], settings['x_max'])  # position (x)
        self.y = uniform(settings['y_min'], settings['y_max'])  # position (y)

        self.r = uniform(0,360)                 # orientation   [0, 360]
        self.v = uniform(0,settings['v_max'])   # velocity      [0, v_max]
        self.dv = uniform(-settings['dv_max'], settings['dv_max'])   # dv

        self.d_food = 100   # distance to nearest food
        self.r_food = 0     # orientation to nearest food
        self.d_superfood = 100
        self.r_superfood = 0
        self.d_poison = 100
        self.r_poison = 0
        self.fitness = 0    # fitness (food count)

        self.wih = wih
        self.who = who

        self.name = name
        
        
    # NEURAL NETWORK
    def think(self):

        # SIMPLE MLP
        af = lambda x: np.tanh(x)               # activation function
        h1 = af(np.dot(self.wih, self.r_superfood))  # hidden layer
        h2 = af(np.dot(self.wih, self.r_food))  # hidden layer
        h3 = af(np.dot(self.wih, self.r_poison))  # hidden layer

        out1 = af(np.dot(self.who, h1))          # output layer
        out2 = af(np.dot(self.who, h2))          # output layer
        out3 = af(np.dot(self.who, h3))          # output layer

        # UPDATE dv AND dr WITH MLP RESPONSE
        self.nn_dv = float(out1[0])   # [-1, 1]  (accelerate=1, deaccelerate=-1)
        self.nn_dr = float(out1[1])   # [-1, 1]  (left=1, right=-1)

        self.nn_dv += float(out2[0])   # [-1, 1]  (accelerate=1, deaccelerate=-1)
        self.nn_dr += float(out2[1])   # [-1, 1]  (left=1, right=-1)

        self.nn_dv += float(out3[0])   # [-1, 1]  (accelerate=1, deaccelerate=-1)
        self.nn_dr += float(out3[1])   # [-1, 1]  (left=1, right=-1)

        
    # UPDATE HEADING
    def update_r(self, settings):
        self.r += self.nn_dr * settings['dr_max'] * settings['dt']
        self.r = self.r % 360

        
    # UPDATE VELOCITY
    def update_vel(self, settings):
        self.v += self.nn_dv * settings['dv_max'] * settings['dt']
        if self.v < 0: self.v = 0
        if self.v > settings['v_max']: self.v = settings['v_max']

    
    # UPDATE POSITION
    def update_pos(self, settings):
        dx = self.v * cos(radians(self.r)) * settings['dt']
        dy = self.v * sin(radians(self.r)) * settings['dt']
        self.x += dx
        self.y += dy


#--- MAIN ---------------------------------------------------------------------+


def run(settings):

    #--- POPULATE THE ENVIRONMENT WITH FOOD ---------------+
    foods = []
    for i in range(0,settings['food_num']):
        foods.append(food(settings))

    #--- POPULATE THE ENVIRONMENT WITH SUPERFOOD ---------------+
    superfoods = []
    for i in range(0,settings['superfood_num']):
        superfoods.append(superfood(settings))    
        
    #--- POPULATE THE ENVIRONMENT WITH POISON ---------------+
    poisons = []
    for i in range(0,settings['poison_num']):
        poisons.append(poison(settings))

    #--- POPULATE THE ENVIRONMENT WITH ORGANISMS ----------+
    organisms = []
    for i in range(0,settings['pop_size']):
        wih_init = np.random.uniform(-1, 1, (settings['hnodes'], settings['inodes']))     # mlp weights (input -> hidden)
        who_init = np.random.uniform(-1, 1, (settings['onodes'], settings['hnodes']))     # mlp weights (hidden -> output)
        
        organisms.append(organism(settings, wih_init, who_init, name='gen[x]-org['+str(i)+']'))
    
    #--- CYCLE THROUGH EACH GENERATION --------------------+
    for gen in range(0, settings['gens']):
        
        # SIMULATE
        organisms = simulate(settings, organisms, foods, superfoods, poisons, gen)

        # EVOLVE
        organisms, stats = evolve(settings, organisms, gen)
        print ('> GEN:',gen,'BEST:',stats['BEST'],'AVG:',stats['AVG'],'WORST:',stats['WORST'])

    pass


#--- RUN ----------------------------------------------------------------------+

run(settings)
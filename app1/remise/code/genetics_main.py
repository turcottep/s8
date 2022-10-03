# Copyright (c) 2018, Simon Brodeur
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA,
# OR PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Université de Sherbrooke
# Code for Artificial Intelligence module
# Adapted by Audrey Corbeil Therrien for Artificial Intelligence module

import time
import genetic
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def train_genetics(
    fitness_function, numparams, min_param_value, max_param_value, target_score
):
    # Fix random number generator seed for reproducible results
    np.random.seed(0)

    # The parameters for encoding the population

    # parameters for the genetic algorithm
    popsize = 2000
    nbits = 8
    numGenerations = 150
    mutationProb = 0.001
    crossoverProb = 0.98

    ga_sim = genetic.Genetic(numparams, popsize, nbits)
    ga_sim.init_pop()
    ga_sim.set_fit_fun(fitness_function)

    ga_sim.set_sim_parameters(
        numGenerations,
        mutationProb,
        crossoverProb,
        min_param_value,
        max_param_value,
    )

    for i in range(ga_sim.num_generations):
        # start timer
        # start_time = time.time()
        ga_sim.decode_individuals()
        # print("time for decoding", time.time() - start_time, "s")

        # start_time = time.time()
        ga_sim.eval_fit()
        # print("time for eval_fit", time.time() - start_time, "s")

        # start_time = time.time()
        best_score = ga_sim.print_progress()
        if best_score >= target_score:
            break
        # print("time for print_progress", time.time() - start_time, "s")

        ga_sim.new_gen()

    # Display best individual
    # print("#########################")
    # print("Best individual (encoded values):")
    # print(ga_sim.get_best_individual())
    # print("#########################")

    # # Display fitness over generations
    # fig = plt.figure()
    # n = np.arange(numGenerations)
    # ax = fig.add_subplot(111)
    # ax.plot(n, ga_sim.maxFitnessRecord, "-r", label="Generation Max")
    # ax.plot(n, ga_sim.overallMaxFitnessRecord, "-b", label="Overall Max")
    # ax.plot(n, ga_sim.avgMaxFitnessRecord, "--k", label="Generation Average")
    # ax.set_title("Fitness value over generations")
    # ax.set_xlabel("Generation")
    # ax.set_ylabel("Fitness value")
    # ax.legend()
    # fig.tight_layout()

    # plt.show()

    # return best individual
    return ga_sim.get_best_individual()

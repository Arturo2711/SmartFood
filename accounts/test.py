from .utility import get_Vector
import random
import numpy as np


def calculate_nutritional_values(chromosome: np.ndarray) -> np.ndarray:
    nutritional_data = [get_Vector(ingredient) for ingredient in chromosome]
    return np.sum(nutritional_data, axis=0)


def calculate_euclidean_distance(source: np.ndarray, target: np.ndarray):
    return np.linalg.norm(source - target)


NUMBER_OF_INGREDIENTS = 1408


def initialize_population(
    population_size: int,
    chromosome_size: int,
    target_nutritional_values: np.ndarray,
) -> list[tuple[list, float]]:
    population = []
    for _ in range(population_size):
        chromosome = np.random.randint(
            1, NUMBER_OF_INGREDIENTS + 1, size=chromosome_size
        )
        chromosome_nutritional_data = calculate_nutritional_values(chromosome)
        chromosome_fitness = calculate_euclidean_distance(
            target_nutritional_values, chromosome_nutritional_data
        )
        population.append((chromosome, chromosome_fitness))
    return population


def tournament_selection(population: list, tournament_size: int = 3):
    selected_contestants = []
    for _ in range(len(population)):
        competitors = random.sample(population, tournament_size)
        winner = min(competitors, key=lambda individual: individual[1])
        selected_contestants.append(winner)
    return selected_contestants


def two_point_crossover(parent1, parent2, target_nutritional_values):
    chromosome_size = len(parent1)
    crossover_point1, crossover_point2 = sorted(
        random.sample(range(1, chromosome_size), 2)
    )

    offspring1 = (
        parent1[:crossover_point1]
        + parent2[crossover_point1:crossover_point2]
        + parent1[crossover_point2:]
    )
    offspring2 = (
        parent2[:crossover_point1]
        + parent1[crossover_point1:crossover_point2]
        + parent2[crossover_point2:]
    )

    fitness_offspring1 = calculate_euclidean_distance(
        target_nutritional_values, calculate_nutritional_values(offspring1)
    )
    fitness_offspring2 = calculate_euclidean_distance(
        target_nutritional_values, calculate_nutritional_values(offspring2)
    )

    return [(offspring1, fitness_offspring1), (offspring2, fitness_offspring2)]


def crossover(population, crossover_rate, target_nutritional_values):
    np.random.shuffle(population)

    offspring = []
    for i in range(0, len(population) - 1, 2):
        if np.random.rand() < crossover_rate:
            child1, child2 = two_point_crossover(
                population[i][0], population[i + 1][0], target_nutritional_values
            )
            offspring.extend([child1, child2])
    return offspring


def mutate(offspring: list, mutation_rate: float):
    for child in offspring:
        chromosome = child[0]
        mutation_mask = np.random.rand(len(chromosome)) < mutation_rate
        mutation_values = np.random.randint(
            1, NUMBER_OF_INGREDIENTS + 1, size=len(chromosome)
        )
        chromosome[mutation_mask] = mutation_values[mutation_mask]
    return offspring


def elitism(population, population_size):
    sorted_population = sorted(population, key=lambda x: x[1])
    elite_individuals = sorted_population[:population_size]
    return elite_individuals


DEFAULT_POPULATION_SIZE = 100
DEFAULT_CROSSOVER_RATE = 0.9
DEFAULT_MUTATION_RATE = 0.5
DEFAULT_NUMBER_OF_GENERATIONS = 100


def genetic(
    target_nutritional_values: list[float],
    chromosome_size: int,
    population_size: int = DEFAULT_POPULATION_SIZE,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    number_of_generations: int = DEFAULT_NUMBER_OF_GENERATIONS,
):
    population = initialize_population(
        population_size, chromosome_size, target_nutritional_values
    )

    for _ in range(number_of_generations):
        selected_population = tournament_selection(population)
        offspring = crossover(
            selected_population, crossover_rate, target_nutritional_values
        )

        mutate(offspring, mutation_rate)

        population = elitism(population + offspring, population_size)

    return population[0][0]

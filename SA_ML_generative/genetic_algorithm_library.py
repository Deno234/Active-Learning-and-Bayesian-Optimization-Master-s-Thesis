import numpy as np


class GeneticAlgorithm:
    class Peptide:
        def __init__(self, sequence, provenance=None):
            self.sequence = sequence
            self.fitness = 0
            self.provenance = provenance or {}

    def __init__(
        self,
        fitness_function,
        similarity_penalty,
        length_penalty,
        min_initial_peptide_length,
        max_initial_peptide_length,
        allowed_amino_acids,
        population_size,
        offspring_count,
        max_num_generations,
        tournament_size,
        mutation_probability,
        population_fitness_function=None,
        event_callback=None,
    ):
        self.fitness_function = fitness_function
        self.similarity_penalty = similarity_penalty
        self.length_penalty = length_penalty
        self.min_initial_peptide_length = min_initial_peptide_length
        self.max_initial_peptide_length = max_initial_peptide_length
        self.allowed_amino_acids = allowed_amino_acids
        self.population_size = population_size
        self.offspring_count = offspring_count
        self.max_num_generations = max_num_generations
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.population_fitness_function = population_fitness_function
        self.event_callback = event_callback
        self.generation = 0

    def _emit(self, event, peptide, **extra):
        if self.event_callback is None:
            return
        self.event_callback(
            {
                "event": event,
                "generation": self.generation,
                "sequence": peptide.sequence,
                "fitness": float(peptide.fitness),
                **peptide.provenance,
                **extra,
            }
        )

    def find_peptides(self):
        population = self.generate_random_population()
        generation = 1

        while True:
            if generation > self.max_num_generations:
                break

            self.generation = generation
            print(f"Generation: {generation}/{self.max_num_generations}")

            self.evaluate_population(population)
            offspring = self.generate_offspring(population)

            population += offspring
            self.evaluate_population(population)

            population = self.next_generation(population)
            generation += 1

        return population

    def generate_random_population(self):
        population = []

        for _ in range(self.population_size):
            sequence = ""
            for _ in range(np.random.randint(self.min_initial_peptide_length, self.max_initial_peptide_length + 1)):
                sequence += self.allowed_amino_acids[np.random.randint(len(self.allowed_amino_acids))]

            peptide = self.Peptide(sequence, {"origin": "initial"})
            population.append(peptide)
            self._emit("created", peptide)

        return population

    def evaluate_population(self, population):
        if self.population_fitness_function is not None:
            fitness_values = self.population_fitness_function(population)
            for peptide, fitness in zip(population, fitness_values):
                peptide.fitness = float(fitness)
                self._emit("evaluated", peptide)
            return

        for peptide in population:
            peptide.fitness = \
                self.fitness_function(peptide.sequence) - \
                self.similarity_penalty(peptide.sequence, population) - \
                self.length_penalty(peptide.sequence)
            self._emit("evaluated", peptide)

    def generate_offspring(self, population):
        offspring = []

        for _ in range(self.offspring_count):
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)

            child = self.recombination(parent1, parent2)
            child = self.mutation(child)

            offspring.append(child)
            self._emit("created", child)

        return offspring

    def tournament_selection(self, population):
        random_parent = population[np.random.randint(len(population))]

        for _ in range(self.tournament_size):
            i = np.random.randint(len(population))
            if population[i].fitness > random_parent.fitness:
                random_parent = population[i]

        return random_parent

    def recombination(self, parent1, parent2):
        p = np.random.rand()
        crossover_index = int(len(parent1.sequence) * p)
        sequence = parent1.sequence[:crossover_index] + parent2.sequence[int(len(parent2.sequence) * p):]

        return self.Peptide(
            sequence,
            {
                "origin": "offspring",
                "parent_1": parent1.sequence,
                "parent_2": parent2.sequence,
                "crossover_fraction": float(p),
                "crossover_index": crossover_index,
                "mutation": "none",
            },
        )

    def mutation(self, child):
        sequence = list(child.sequence[:])

        if np.random.rand() < self.mutation_probability:
            r = np.random.rand()

            if 0 <= r < 0.25:
                position = np.random.randint(len(sequence) + 1)
                amino_acid = self.allowed_amino_acids[
                    np.random.randint(len(self.allowed_amino_acids))
                ]
                sequence.insert(position, amino_acid)
                child.provenance["mutation"] = f"insert:{position}:{amino_acid}"
            elif 0.25 <= r < 0.5:
                if len(sequence) > 1:
                    first_position, second_position = np.random.randint(len(sequence), size=2)
                    sequence[first_position], sequence[second_position] = \
                        sequence[second_position], sequence[first_position]
                    child.provenance["mutation"] = (
                        f"swap:{first_position}:{second_position}"
                    )
            elif 0.5 <= r < 0.75:
                if len(sequence) > 1:
                    position = np.random.randint(len(sequence))
                    del sequence[position]
                    child.provenance["mutation"] = f"delete:{position}"
            if 0.75 <= r <= 1:
                position = np.random.randint(len(sequence))
                amino_acid = self.allowed_amino_acids[
                    np.random.randint(len(self.allowed_amino_acids))
                ]
                sequence[position] = amino_acid
                child.provenance["mutation"] = (
                    f"substitute:{position}:{amino_acid}"
                )

        sequence = "".join(sequence)

        return self.Peptide(sequence)

    def next_generation(self, population):
        sorted_population = sorted(population, key=lambda peptide: peptide.fitness)
        survivors = sorted_population[-self.population_size:]
        survivor_ids = {id(peptide) for peptide in survivors}
        for peptide in population:
            self._emit(
                "survival",
                peptide,
                survived=id(peptide) in survivor_ids,
            )
        return survivors

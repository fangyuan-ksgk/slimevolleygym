#!/usr/bin/env python
"""
Train an agent to play SlimeVolley using NEAT-Python with tournament-style evaluation.
"""
import os
import gym
import neat
import numpy as np
import random
from collections import defaultdict
import pickle
import slimevolleygym
from slimevolleygym import SlimeVolleyEnv

def eval_genome_vs_opponent(genome, opponent_genome, config):
    """Evaluate a genome against an opponent in the SlimeVolley environment"""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    opponent_net = neat.nn.FeedForwardNetwork.create(opponent_genome, config)
    
    env = gym.make("SlimeVolley-v0", disable_env_checker=True)
    env.survival_bonus = True  # Enable survival bonus
    env = env.unwrapped  # Access raw environment for multiagent mode
    
    total_fitness = 0
    wins = 0
    losses = 0
    
    for episode in range(3):  # Multiple episodes for robust evaluation
        obs = env.reset()
        opponent_obs = env.obs_agent_two()
        episode_fitness = 0
        done = False
        
        while not done:
            # Get actions from neural networks
            action_values = net.activate(obs)
            action = [1 if v > 0.5 else 0 for v in action_values]
            
            opponent_action_values = opponent_net.activate(opponent_obs)
            opponent_action = [1 if v > 0.5 else 0 for v in opponent_action_values]
            
            # Take actions in environment
            obs, reward, done, info = env.step(action, opponent_action)
            episode_fitness += reward
            opponent_obs = info['otherState']
            
            if done:
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                break
    
    env.close()
    total_fitness = episode_fitness + (wins - losses)
    return total_fitness, wins, losses

def tournament_evaluation(population, config):
    """Evaluate genomes through tournament-style matches"""
    scores = defaultdict(float)
    wins_record = defaultdict(int)
    matches_played = defaultdict(int)
    winning_streaks = defaultdict(int)
    
    # Tournament rounds
    for _ in range(len(population) * 2):
        genome1_id, genome2_id = random.sample(list(population.keys()), 2)
        genome1, genome2 = population[genome1_id], population[genome2_id]
        
        fitness, g1_wins, g1_losses = eval_genome_vs_opponent(genome1, genome2, config)
        
        scores[genome1_id] += fitness
        wins_record[genome1_id] += g1_wins
        matches_played[genome1_id] += 1
        
        # Update winning streaks
        if g1_wins > g1_losses:
            winning_streaks[genome1_id] += 1
        else:
            winning_streaks[genome1_id] = 0
    
    # Calculate final fitness scores
    for genome_id in population:
        if matches_played[genome_id] > 0:
            base_fitness = scores[genome_id] / matches_played[genome_id]
            streak_bonus = winning_streaks[genome_id] * 0.1
            population[genome_id].fitness = base_fitness + streak_bonus
            population[genome_id].winning_streak = winning_streaks[genome_id]
        else:
            population[genome_id].fitness = 0
            population[genome_id].winning_streak = 0

def eval_genomes(genomes, config):
    """Evaluate all genomes in the population"""
    # Convert genomes to dict for tournament evaluation
    population = {genome_id: genome for genome_id, genome in genomes}
    tournament_evaluation(population, config)

def run_neat(config_file):
    """Run NEAT algorithm with tournament-style evaluation"""
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)

    # Create population
    pop = neat.Population(config)

    # Add reporters for detailed statistics
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(CustomReporter())
    
    # Save checkpoint every 10 generations
    pop.add_reporter(neat.Checkpointer(10, filename_prefix='neat-checkpoint-'))

    # Run for up to 1000 generations to match GA's computational scale
    winner = pop.run(eval_genomes, 1000)

    # Save the winner
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print('\nBest genome:\n{!s}'.format(winner))
    return winner

class CustomReporter(neat.reporting.BaseReporter):
    """Custom reporter for detailed statistics"""
    def post_evaluate(self, config, population, species, best_genome):
        """Print detailed statistics after each generation"""
        print(f"\nGeneration Statistics:")
        print(f"Best Fitness: {best_genome.fitness:.6f}")
        print(f"Number of Species: {len(species.species)}")
        print(f"Population Size: {len(population)}")
        print(f"Best Genome Size: {len(best_genome.connections)}")
        print(f"Best Genome Key: {best_genome.key}")
        print(f"Winning Streak: {getattr(best_genome, 'winning_streak', 0)}")
        print(f"Species Sizes: {[len(s.members) for s in species.species.values()]}")

if __name__ == '__main__':
    # Get local directory
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat')
    
    winner = run_neat(config_path)

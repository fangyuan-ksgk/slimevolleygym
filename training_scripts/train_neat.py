#!/usr/bin/env python
"""
Train an agent to play SlimeVolley using NEAT-Python.
"""
import os
import gym
import neat
import random
import numpy as np
import slimevolleygym
from slimevolleygym import SlimeVolleyEnv

def eval_genome_vs_opponent(genome, opponent_genome, config, num_episodes=3):
    """
    Evaluate a genome against a specific opponent genome using the environment's multiagent mode.
    Returns the fitness score based on match outcomes and episode rewards.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    opponent_net = neat.nn.FeedForwardNetwork.create(opponent_genome, config)
    
    env = gym.make("SlimeVolley-v0", disable_env_checker=True, new_step_api=True)
    env.survival_bonus = True  # Enable survival bonus for better training signal
    env = env.unwrapped  # Unwrap to access raw environment for multiagent mode
    
    total_fitness = 0
    wins = 0
    losses = 0
    
    for episode in range(num_episodes):
        episode_fitness = 0
        obs = env.reset()
        opponent_obs = env.game.agent_left.getObservation()  # Get initial observation for opponent
        done = False
        
        while not done:
            # Get actions from both networks
            action_values = net.activate(obs)
            action = [1 if v > 0.5 else 0 for v in action_values]
            
            opponent_action_values = opponent_net.activate(opponent_obs)
            opponent_action = [1 if v > 0.5 else 0 for v in opponent_action_values]
            
            # Take actions in environment using raw environment for multiagent mode
            obs, reward, done, info = env.step(action, opponent_action)
            episode_fitness += reward
            
            # Get opponent's observation for next step
            opponent_obs = info['otherState']
            
            # Track game outcomes based on lives
            if done:
                if info['ale.lives'] > info['ale.otherLives']:
                    wins += 1
                elif info['ale.lives'] < info['ale.otherLives']:
                    losses += 1
        
        total_fitness += episode_fitness
    
    env.close()
    
    # Calculate fitness incorporating wins and episode rewards
    avg_fitness = total_fitness / num_episodes
    if wins + losses > 0:
        win_ratio = wins / (wins + losses)
        avg_fitness += win_ratio * 2  # Scale win ratio to be significant
        
        # Add winning streak bonus similar to GA implementation
        if wins > 2:  # Bonus for winning all episodes
            avg_fitness += wins * 0.5  # Additional bonus for winning streaks
    
    return avg_fitness, wins, losses

def tournament_evaluation(population, config, tournament_size=4):
    """
    Conduct a tournament-style evaluation where genomes compete against each other.
    Returns a dictionary of genome keys to their tournament performance scores.
    Also tracks winning streaks similar to GA implementation.
    """
    scores = {genome.key: 0 for genome in population}
    wins = {genome.key: 0 for genome in population}
    matches = {genome.key: 0 for genome in population}
    winning_streaks = {genome.key: getattr(genome, 'winning_streak', 0) for genome in population}
    
    # Generate tournament brackets
    genomes = list(population)
    random.shuffle(genomes)
    
    # Run tournament matches
    for i in range(0, len(genomes), 2):
        if i + 1 >= len(genomes):
            break
            
        genome1 = genomes[i]
        genome2 = genomes[i + 1]
        
        fitness, g1_wins, g1_losses = eval_genome_vs_opponent(genome1, genome2, config)
        
        # Update scores for both genomes
        scores[genome1.key] += fitness
        scores[genome2.key] += -fitness  # Opponent's perspective
        
        # Update wins and matches
        wins[genome1.key] += g1_wins
        wins[genome2.key] += g1_losses
        matches[genome1.key] += g1_wins + g1_losses
        matches[genome2.key] += g1_wins + g1_losses
        
        # Update winning streaks similar to GA
        if g1_wins > g1_losses:
            winning_streaks[genome1.key] = winning_streaks[genome2.key] + 1
            winning_streaks[genome2.key] = 0
        elif g1_losses > g1_wins:
            winning_streaks[genome2.key] = winning_streaks[genome1.key] + 1
            winning_streaks[genome1.key] = 0
        
        # Store winning streaks in genomes for persistence
        genome1.winning_streak = winning_streaks[genome1.key]
        genome2.winning_streak = winning_streaks[genome2.key]
    
    return scores, wins, matches, winning_streaks

def eval_genome(genome, config):
    """
    Evaluate a single genome using tournament-style evaluation and previous best.
    Incorporates winning streak bonuses similar to GA implementation.
    """
    global pop
    
    if not hasattr(eval_genome, "generation_best"):
        eval_genome.generation_best = None
    
    if not hasattr(genome, 'winning_streak'):
        genome.winning_streak = 0
    
    total_fitness = 0
    total_wins = 0
    total_matches = 0
    
    # First, evaluate against previous best if available
    if eval_genome.generation_best is not None:
        fitness, wins, losses = eval_genome_vs_opponent(genome, eval_genome.generation_best, config)
        total_fitness += fitness * 2  # Weight matches against champion more heavily
        total_wins += wins
        total_matches += wins + losses
        
        # Update winning streak against champion
        if wins > losses:
            genome.winning_streak = getattr(eval_genome.generation_best, 'winning_streak', 0) + 1
        elif losses > wins:
            genome.winning_streak = 0
    
    # Then participate in tournament
    population = [g for g in pop.population.values() if g.key != genome.key]
    tournament_opponents = random.sample(population, min(4, len(population)))
    tournament_population = tournament_opponents + [genome]
    
    scores, wins, matches, winning_streaks = tournament_evaluation(tournament_population, config)
    
    # Add tournament results to total
    total_fitness += scores[genome.key]
    total_wins += wins[genome.key]
    total_matches += matches[genome.key]
    
    # Calculate final fitness with tournament performance and winning streak
    if total_matches > 0:
        avg_fitness = total_fitness / (total_matches / 3)  # Normalize by number of episodes
        win_ratio = total_wins / total_matches
        avg_fitness += win_ratio * 2  # Additional bonus for winning
        
        # Add winning streak bonus similar to GA
        if genome.winning_streak > 0:
            avg_fitness += genome.winning_streak * 0.5  # Bonus scales with streak length
        
        # Bonus for dominating tournament
        if wins[genome.key] == max(wins.values()):
            avg_fitness += 1.0  # Tournament winner bonus
    else:
        avg_fitness = 0
    
    return avg_fitness

def eval_genomes(genomes, config):
    """
    Evaluate all genomes in the population.
    """
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run_neat(config_file):
    """
    Run NEAT algorithm to train an agent using self-play.
    """
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)

    # Create the population and make it globally accessible
    global pop
    pop = neat.Population(config)

    # Add reporters to show progress
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    # Save checkpoint every 10 generations
    pop.add_reporter(neat.Checkpointer(10, filename_prefix='neat-checkpoint-'))

    def post_evaluate_callback(config, population, species_set, best_genome):
        """Update the generation's best performer after evaluation"""
        eval_genome.generation_best = best_genome
        
        # Print detailed statistics for monitoring
        print(f"\nGeneration Statistics:")
        print(f"Best Fitness: {best_genome.fitness:.6f}")
        print(f"Number of Species: {len(species_set.species)}")
        print(f"Population Size: {len(population)}")
        print(f"Best Genome Size: {len(best_genome.connections)}")
        print(f"Best Genome Key: {best_genome.key}")
        print(f"Winning Streak: {getattr(best_genome, 'winning_streak', 0)}")
    
    # Create and add custom reporter with post-evaluation callback
    class CustomReporter(neat.reporting.BaseReporter):
        def post_evaluate(self, config, population, species, best_genome):
            post_evaluate_callback(config, population, species, best_genome)
    
    pop.add_reporter(CustomReporter())
    
    # Run for up to 1000 generations to match GA's tournament count
    # GA does 500k tournaments with 128 agents ≈ 3900 tournaments per agent
    # Our system does ~4 matches per genome per generation, so we need ~975 generations
    winner = pop.run(eval_genomes, 1000)  # Increased to match GA's computational scale

    # Save the winner
    with open('winner-feedforward', 'wb') as f:
        import pickle
        pickle.dump(winner, f)

    print('\nBest genome:\n{!s}'.format(winner))
    return winner

if __name__ == '__main__':
    # Get local directory
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat')
    
    winner = run_neat(config_path)

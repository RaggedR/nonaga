"""Hyperparameters for AlphaZero training."""


class Config:
    # --- MCTS ---
    num_mcts_sims = 200         # MCTS simulations per move (was 100, deeper search for better policy targets)
    cpuct = 1.5                 # Exploration constant
    dirichlet_alpha = 0.15      # Dirichlet noise parameter (was 0.3, spikier noise forces more exploration)
    dirichlet_epsilon = 0.35    # Weight of Dirichlet noise at root (was 0.25, more noise mixed in)

    # --- Self-play ---
    num_self_play_games = 100   # Games per iteration
    temp_threshold = 30         # Plies before switching to low temp
    temp_late = 1.0             # Temperature after threshold (keep stochastic for decisive games)
    max_game_plies = 500        # Max plies before declaring draw

    # --- Training ---
    num_iterations = 100        # Total training iterations
    num_epochs = 2              # Epochs per training step (was 5, reduced for speed)
    batch_size = 128            # (was 64, larger for faster training)
    learning_rate = 0.001
    weight_decay = 1e-4
    replay_buffer_size = 8      # Keep last N iterations of data (was 3, preserves opening diversity)

    # --- Arena ---
    arena_games = 20            # Games to compare new vs old (was 10, less noisy acceptance)
    arena_threshold = 0.50      # Score to accept new model (draws=0.5)

    # --- Parallel self-play ---
    num_workers = 8              # Parallel self-play workers

    # --- Curriculum pretraining ---
    curriculum_pretrain_iters = 20   # Iterations with adjacency win
    curriculum_num_mcts_sims = 25    # Fewer sims (simpler game)
    curriculum_num_games = 50        # Games per pretraining iteration
    curriculum_win_mode = 'adjacency'  # Win condition during pretraining

    # --- Island-model AlphaZero ---
    num_islands = 5
    island_migration_freq = 5       # iterations between migrations
    island_migration_rate = 0.1     # fraction of replay buffer to share
    island_cross_play_rate = 0.3    # fraction of games that are cross-play with ring neighbor
    island_topology = 'ring'

    # --- Checkpoints ---
    checkpoint_dir = "checkpoints"

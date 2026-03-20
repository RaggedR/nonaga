"""Hyperparameters for AlphaZero training."""


class Config:
    # --- MCTS ---
    num_mcts_sims = 100         # MCTS simulations per move
    cpuct = 1.5                 # Exploration constant
    dirichlet_alpha = 0.3       # Dirichlet noise parameter
    dirichlet_epsilon = 0.25    # Weight of Dirichlet noise at root

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
    replay_buffer_size = 3      # Keep last N iterations of data (was 5)

    # --- Arena ---
    arena_games = 10            # Games to compare new vs old
    arena_threshold = 0.50      # Score to accept new model (draws=0.5)

    # --- Parallel self-play ---
    num_workers = 8              # Parallel self-play workers

    # --- Checkpoints ---
    checkpoint_dir = "checkpoints"

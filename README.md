# Ocean Multi-Agent Simulation with CasADi Optimization

A MATLAB simulation system for multi-agent navigation in dynamic ocean environments using CasADi optimization for trajectory planning. The system provides realistic ocean current modeling, formation control, and efficient trajectory optimization with real-time visualization.

## Features

### 🌊 Ocean Current Modeling
- **Lamb-Oseen Vortex Models**: Multiple vortices with configurable strength and core radius
- **Time-Varying Currents**: Dynamic ocean environments with temporal variations
- **Vectorized Computation**: Efficient current evaluation for multiple positions simultaneously
- **Noise Modeling**: Realistic current estimation with configurable noise levels
- **CasADi Integration**: Symbolic ocean functions for optimization

### 🚁 Multi-Agent Planning
- **CasADi Optimization**: Nonlinear trajectory planning using direct transcription
- **Formation Control**: Configurable geometric formations (triangle, square, line)
- **Obstacle Avoidance**: Constraint-based avoidance of circular obstacles
- **Collision Avoidance**: Inter-agent safety with configurable margins
- **Control Limits**: Respect maximum agent speed constraints

### 🎯 System Capabilities
- **Real-Time Visualization**: Live display of currents, agents, paths, and plans
- **Flexible Configuration**: Comprehensive parameter system with validation
- **Performance Optimized**: Vectorized operations and efficient algorithms
- **Robust Error Handling**: Graceful fallbacks when optimization fails

## Directory Structure

```
ocean-new/
├── main_simulation.m              # Main simulation script
├── setup_paths.m                  # CasADi path configuration
├── config/
│   └── default_config.m          # Configuration management and validation
├── core/
│   ├── agents/
│   │   ├── initialize_agents.m   # Agent initialization (random/formation)
│   │   └── update_agent_state.m  # Agent state dynamics
│   ├── ocean_models/
│   │   ├── calculate_ocean_current_vectorized.m  # Vectorized current computation
│   │   ├── get_noisy_current_estimate.m          # Realistic current estimation
│   │   ├── create_symbolic_ocean_func.m          # CasADi function generation
│   │   └── [other ocean modeling functions]
│   └── planning/
│       ├── casadi_multi_agent_planner.m          # Main trajectory optimizer
│       └── user_planning_algorithm_placeholder.m # Extensible planner interface
├── visualization/
│   ├── plot_environment.m        # Environment and agent visualization
│   ├── create_circle_vertices.m  # Geometric utilities
│   └── cmap.m                    # Custom colormaps
└── utils/
    └── casadi-users_guide-v3.7.0.pdf
```

## Quick Start

### Prerequisites
- MATLAB R2018a or later (R2020a+ recommended)
- CasADi 3.5+ (path configured in `setup_paths.m`)

### Basic Usage

1. **Configure CasADi Path**: Edit `setup_paths.m` with your CasADi installation path
2. **Run Simulation**: Execute `main_simulation.m` directly

```matlab
% Run main_simulation.m for default simulation
main_simulation
```

### Custom Configuration

Edit parameters directly in `main_simulation.m` or use the configuration system:

```matlab
% Load and modify default configuration
config = default_config();
config.agents.count = 6;                    % Number of agents
config.simulation.total_time = 300;         % Simulation duration (s)
config.simulation.enable_formation = true;  % Enable formation control
config.ocean.type = 'time_varying';         % Dynamic currents

% Apply configuration (integrate with main_simulation.m)
```

## Configuration Options

### Simulation Parameters
| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| `dt` | Time step | 0.5 | s |
| `T_final` | Total simulation time | 500 | s |
| `visualization` | Enable real-time display | true | - |
| `formation_enabled` | Enable formation control | true | - |

### Agent Parameters
| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| `num_agents` | Number of agents | 4 | - |
| `radius` | Agent physical radius | 1.5 | m |
| `max_speed` | Maximum control speed | 5.0 | m/s |
| `formation_inter_agent_distance` | Formation spacing | 8.0 | m |

### Ocean Parameters
| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| `type` | Current model | 'static' | - |
| `noise_level` | Current estimation noise | 0.0 | m/s |
| `gradient_noise_level` | Gradient estimation noise | 0.0 | 1/s |

### Environment Parameters
| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| `x_limits` | Environment X bounds | [-50, 50] | m |
| `y_limits` | Environment Y bounds | [-50, 50] | m |
| `obstacles` | Circular obstacles | [] | - |

## Key Components

### Ocean Current Model
- **Vortex Superposition**: Multiple Lamb-Oseen vortices with time variation
- **Vectorized Evaluation**: Efficient computation for multiple query points
- **Symbolic Functions**: CasADi integration for optimization compatibility

### Multi-Agent Planner
- **Direct Transcription**: Position-based trajectory optimization
- **Constraint Handling**: Obstacles, boundaries, speed limits, formation
- **Robust Solving**: IPOPT solver with error handling and fallbacks

### Visualization System
- **Current Field Display**: Contour plots and vector fields
- **Agent Tracking**: Real-time position, velocity, and plan visualization
- **Formation Links**: Visual formation connectivity (when enabled)

## Usage Examples

### Add Obstacles
```matlab
% Edit in main_simulation.m
env_params.obstacles(1) = struct('center', [0; 0], 'radius', 15);
env_params.obstacles(2) = struct('center', [20; -10], 'radius', 8);
```

### Enable Time-Varying Currents
```matlab
% Edit in main_simulation.m
current_params.type = 'time_varying';
current_params.noise_level = 0.1;
```

### Adjust Formation
```matlab
% Edit in main_simulation.m
sim_params.formation_enabled = true;
agent_params.formation_inter_agent_distance = 10.0;
agent_params.formation_weight = 1.0;
```

## Performance Tips

1. **Reduce Visualization Frequency**: Increase `vis_interval` for faster simulation
2. **Optimize Planning Horizon**: Shorter horizons solve faster but may be suboptimal  
3. **Disable Formation**: Set `formation_enabled = false` for simpler optimization
4. **Adjust Solver Tolerance**: Balance accuracy vs. speed in CasADi settings

## Troubleshooting

### CasADi Issues
- Verify CasADi path in `setup_paths.m`
- Check MATLAB/CasADi version compatibility
- Ensure CasADi installation includes IPOPT solver

### Planning Failures
- Reduce planning horizon for complex scenarios
- Check constraint feasibility (obstacles, boundaries)
- Increase solver iterations or tolerance

### Visualization Problems
- Close figure windows if simulation continues running
- Reduce visualization grid resolution for performance
- Check graphics drivers for rendering issues

## Dependencies

- **MATLAB**: R2018a+ (R2020a+ recommended)
- **CasADi**: 3.5+ with IPOPT solver
- **Optional**: Parallel Computing Toolbox (for potential speedups)

## License

Academic research use. Please cite appropriately in publications. 
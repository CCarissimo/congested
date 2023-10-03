# congested: learning in congestion games

what are the features of the package that are most important?
- the many implemented environments specific to game theory
- the implementation of Q-learning with easy extensions of its variants
- internal knowledge of expected game Nash Equilibria
- standard parameters implemented as defaults of the games

## steps to make this a package
- use pip requirements, and make sure only the necessary requirements are present
- use dataclasses to achieve modularity, where possible, needed for:
  - run simulation code to know which parameters it needs, e.g.
    - duopoly requires states but congestion does not
    - public goods game takes a multiplier
    - network congestion games take parameters for each edge
  - multiprocessing sweeps to know necessary parameters
  - plotting of relevant variables and parameters after simulation
- documentation of important functions
- plotting functions
  - welfare over time
  - action distribution over time
  - vector field plot, simplex
  - q values plots
- setuptools toml file
- package structure



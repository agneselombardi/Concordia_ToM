# Concordia_ToM

Code and Dataset from "Doing Things with Words: Rethinking Theory of Mind Simulation in Large Language Models"

# Abstract
Language is fundamental to human cooperation, enabling not only the exchange of information but also the coordination of actions through shared interpretations of context. This study investigates whether the Generative Agent-Based Model (GABM) Concordia can effectively simulate Theory of Mind (ToM) in realistic environments. We evaluate if this framework accurately models ToM abilities and if GPT-4 can perform tasks by genuinely inferring social context rather than relying on linguistic memorization.

Our results reveal a significant limitation: GPT-4 often fails to select actions based on belief attribution, indicating that previously observed ToM-like abilities may arise from shallow statistical associations rather than true reasoning. Additionally, the model struggles to generate coherent causal effects from agent actions, exposing challenges in handling complex social interactions. These findings challenge current claims about emergent ToM capabilities in LLMs and highlight the need for more rigorous, action-based evaluation methods.

# Usage
Run the `/queries/theory_of_mind_simulation.ipynb` notebook to create the simulation and obtain results for all tasks. The `/queries` folder contains HTML files with simulation results.

The original dataset is available in the `/data` folder in both `.xlsx` and `.csv` formats, organized by task stimuli (1 to 5).

The `/analysis/evaluation.ipynb` notebook contains code for generating graphs, while `/analysis/ratings.ipynb` contains code to generate ratings.

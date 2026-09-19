# Nordic AI Cup 2026: Survival Simulator

## Who I am and how to work with me

I am Arnaud, a few weeks into a master's in data science for business at BI Oslo.
My coding background is 1 math class for data science and 1 class on numerical
methods in Python. I also know C, C++, Matlab, Java and Mathematica at a basic to
intermediate level. I have no machine learning background yet.

**My goal here is to learn, not to win.** Treat this repository as a guided
exercise. A working solution I understand is worth more to me than a better one I
do not.

### How to explain things to me

- Go one short step at a time, roughly a sentence or two, then stop and check
  whether I have understood before continuing. Long multi-part explanations
  delivered all at once are too much for me to absorb.
- When the answer is simple, keep it very short. Bullet points, 5 maximum, no
  long build-up. Match the length of the reply to how simple the question is.
- Write numbers as numerals, not spelled out in words.
- Never use an en dash or an em dash in anything you write. Write a full
  sentence instead.
- I will ask basic questions. Answer them plainly and without condescension.

### How to work on the code

- Default to explaining before editing. If I have not asked you to change a
  file, do not change it.
- Use plan mode when the change is more than a few lines, so I can see the
  reasoning before the diff.
- When you do write code, explain what each block does in plain language
  afterwards, in the same short-step style.
- Prefer simple, readable Python over clever Python. I need to be able to read
  it back in 6 months.
- If you use a concept I have probably not met yet, say so explicitly and offer
  to explain it rather than assuming.

## The project

This is my own copy of the survival-simulator challenge from the Nordic AI Cup
2026, hosted by Ambolt AI across Sweden, Denmark, Finland, Norway and Iceland.

Upstream repository: https://github.com/amboltio/Nordic-AI-Cup-2026

### What the challenge is

I control an entire species of herbivores as a hivemind. The simulation ticks
every 0.1 simulated seconds, up to a maximum of 3000 simulated seconds. At each
tick I receive, for every living agent, its status and its observations. I reply
with 1 action per agent.

Agent status fields: agent_id, observations, energy, biome, age, speed,
sprint_speed, hearing_radius, vision_angle, vision_range, max_energy. The last 6
are static traits that can mutate when a new agent is spawned.

Observations come from a vision cone plus a hearing and smell radius. Walls and
edges can only be seen, never heard. Observation types and their data:

| Type      | Data                                                        |
|-----------|-------------------------------------------------------------|
| Fruit     | type, distance, angle (radians)                             |
| Agent     | type, distance, angle, relative looking direction           |
| Predator  | type, distance, angle, relative looking direction           |
| Tree      | type, distance, angle                                       |
| Edge      | type, coordinates (start, end)                              |

My action per agent: agent_id, move_distance (float), move_direction (absolute,
radians), turn_angle (radians), spawn_agent (boolean).

### Energy costs

| Action                                          | Cost                                          |
|-------------------------------------------------|-----------------------------------------------|
| Walking, move_distance <= speed                 | move_distance * 0.05                          |
| Sprinting, speed < move_distance <= sprint_speed| speed * 0.05 + (move_distance - speed) * 0.5  |
| Turning                                         | abs(turn_angle) / (2 * pi)                    |
| Spawning a new agent                            | 100                                           |
| Passive living cost                             | 0.1 * biome_energy_modifier                   |

After a random age between 60 and 120, the living cost rises by 0.01 * age.
Sprinting costs 10 times as much per unit distance as walking. That trade-off is
the heart of the problem.

### Scoring

Score is mainly how long the species survives out of 3000 simulated seconds.
Eating fruit adds a small amount. Getting eaten by a predator subtracts an amount
based on the energy that agent still had. Predators spawn over time with rising
probability and kill any agent they touch. 5 agents spawn at the start.

### Baseline to beat

The shipped dummy policy scored 21.16, with every agent dead after 21 simulated
seconds out of 3000. That is the floor.

## Key files

| File                                          | What it is                            |
|-----------------------------------------------|---------------------------------------|
| `local_playground.py`                         | Runs a simulation locally. Set verbose=True for a pygame window. |
| `src/utils/controllers/dummy_agent_policy.py` | The default logic. This is the file to replace. |
| `src/utils/DTOs.py`                           | The data structures.                  |
| `agent_server.py`                             | Serves my policy as a FastAPI endpoint on port 9052. |
| `simulation_server.py`                        | Runs a simulation against that endpoint. |
| `src/core.py`                                 | The simulation core.                  |

## Setup

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python local_playground.py
```

Run headless and fast for testing:

```bash
python -c "from local_playground import local_simulation; local_simulation(verbose=False)"
```

## Submission mechanics

Everything goes through cases.nordicaicup.com with my team API key. My agent has
to run as a web server the evaluation service can reach from the internet, so
either a cloud VM with the port open, or port forwarding from my laptop, which
often fails on a university network.

Validation attempts are unlimited and use random seeds. Evaluation is 1 attempt
only, runs 3 simulations in a row, and the score is the average, so the server
has to stay up through all 3. The server waits at most 10 seconds for a response,
and if accumulated wait reaches 600 seconds the run ends. The simulation is
deterministic only on the same operating system, so a seed only reproduces on
Linux.

## Working notes

Keep a running log of what we tried and what it scored, so I can see the
progression rather than just the final answer. Add it below this line.

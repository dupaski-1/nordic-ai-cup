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

## Git

Upstream repository: https://github.com/amboltio/Nordic-AI-Cup-2026
Do not use git worktrees. Work directly in this folder on the current branch.

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

My action per agent: agent_id, move_distance (float), move_direction (relative to
the agent's current facing direction, radians), turn_angle (radians), spawn_agent
(boolean). The README calls move_direction absolute, but the code adds it to the
agent's current direction, so it is actually relative.

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

Python 3.10 or newer. Use the existing `nordicai` conda environment, do not
create a venv or any other environment.

```bash
conda activate nordicai
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
and if accumulated wait reaches 1200 seconds the run ends. The simulation is
deterministic only on the same operating system, so a seed only reproduces on
Linux.

## Working notes

Keep a running log of what we tried and what it scored, so I can see the
progression rather than just the final answer. Add it below this line.

- Dummy baseline: 21.16, all agents dead at 21s.
- Ring formation policy v1 (clustering, fruit-seeking, wall avoidance,
  spawn once energy > 100), not yet committed: 31.42 on one local run
  (seed 2300228456). Observed bug: agents visibly got stuck in corners
  and stopped moving, and often ended up facing a wall or straight out
  of the map when close to the boundary.
- Ring formation policy v2: fixed 3 wall-avoidance gaps.
  1. A lone/searching agent (no clustermates, no fruit, nobody to join)
     was walking straight ahead every tick with no wall check at all.
  2. Merging into a cluster called the movement function without passing
     it the wall data, so wall avoidance silently got skipped right at
     that moment.
  3. The facing-correction only checked the first nearby wall in view,
     so in a corner (two walls close together) fixing away from one
     could still leave the agent staring at the other, or through the
     gap past the corner. Now checks all nearby walls and retries up to
     a full turn.
  Scores on 4 local runs: 42.62, 41.85, 30.93, 22.06 (different random
  seeds each run). Better on most seeds than v1, but still a lot of
  variance and one run barely beat baseline, so walls were not the only
  cause of early deaths.
- Watched v2 run live (verbose=True, seed 3672530907): scored 202.35, last
  agent dead at 196.6s. Confirms the wall fixes work, not just on paper.
- Still saw agents standing still for a long time in that run, sometimes
  near a tree. Traced it with a temporary debug print: not trees (they
  don't block movement in this game), and not the map's random obstacle
  either. It was `_avoid_walls`, the function that redirects a move away
  from a nearby wall/obstacle: it used a crude "blocked" test (any
  direction with even a loose component toward the wall counts as
  blocked, a full half-circle per wall), while the facing-avoidance
  function next to it already used a narrower, correct test. Near a
  corner the crude test could rule out all 4 directions it tried and the
  agent would stand there permanently, since obstacles never move.
  Debug run before the fix: 157 near-zero-move ticks, 118 of them one
  agent stuck back to back for 11.8s straight.
- Ring formation policy v3: rewrote `_avoid_walls` to use the same
  narrow-cone test as the facing function, via one shared helper
  (`_find_clear_angle`) both now call, so they can't drift apart again.
  Re-ran the same debug check after: 0 genuinely-boxed-in cases across 3
  runs, so the corner-stuck bug looks fixed for real (the earlier count
  was conflating it with agents intentionally holding a ring position
  near a wall, which isn't a bug). Scores on 3 runs: 46.34, 28.42, 230.87.
- Watched v3 live: still saw an agent get stuck, this time squeezed
  between the map's random obstacle and the boundary wall. v3's
  angle-cone test doesn't shrink or grow with how close a wall actually
  is, so it can still misjudge a narrow gap.
- Ring formation policy v4: replaced the movement check with a real
  geometric one. `_avoid_walls` now samples points along the actual
  straight-line path a move would take and checks their distance to the
  true wall/obstacle edges (not just a fixed angle cone), redirecting to
  the nearest angle whose path stays clear. (Facing still uses the old
  angle-cone test, since it's just a gaze heuristic, not a real move.)
  First pass used a 10-unit safety clearance and still found agents
  genuinely stuck 10-30s straight in every one of 5 test runs, this
  time for real (confirmed with a debug counter, not a false positive).
  Turned out the clearance was more conservative than the game itself:
  agent radius is only 5, so 10 was ruling out gaps the game would
  actually let an agent through. Dropped it to 6: 0 genuinely-stuck
  cases across 10 runs. Scores on those 10 runs: 147.9, 169.5, 33.9,
  32.0, 88.8, 67.9, 154.7, 26.9, 102.3, 150.2 (average ~97). Debug code
  removed after confirming.
- Watched v4 live: still got stuck, this time between the map's random
  obstacle and the boundary wall. Turns out there isn't one random
  obstacle, there are 80 (`env_width // 20` in src/utils/simulation.py,
  a detail I got wrong earlier). Wrote a standalone ground-truth tracer
  (reads real agent x,y from sim.env directly, not through the policy's
  limited observations) to confirm: one agent was genuinely wedged
  among 3 clustered obstacles, net movement of 1-4 units over rolling
  3-second windows, for 38 real seconds straight (t=71 to t=109 in one
  run), not a false alarm.
- Ring formation policy v5: two additions to `_avoid_walls`.
  1. Per-agent "which way am I deflecting" memory (`_avoid_bias`). Once
     an agent starts going around an obstacle turning e.g. left, it
     keeps preferring left instead of the one-step sweep re-deciding
     fresh each tick, which was causing left-right-left zigzagging in
     place near tight obstacle clusters (classic "wall following").
     Helped (avg score across 5 runs ~97 -> better spread) but didn't
     fully solve genuine 3-obstacle pockets, since it's still only a
     one-step lookahead.
  2. Stall-escape fallback (`_stall_ticks`, `_retreat_from`): if an
     agent has spent 20+ ticks (2s) in a row actively deflecting with
     no real escape, stop maneuvering cleverly and just retreat
     straight away from the combined center of every nearby wall/
     obstacle edge. Close to guaranteed to clear a tight pocket, even
     if not the shortest way out. Re-ran the tracer on the exact seed
     with the 38s freeze: no more sustained streaks anywhere, just
     brief 2-4s episodes scattered as the agent actually travels and
     occasionally gets briefly caught elsewhere. Scores on 8 fresh
     runs: 95.9, 223.8, 210.3, 145.1, 27.8, 61.5, 195.7, 194.8
     (average ~144, up from ~97).
- Watched v5 live (seed 3346480106): scored 105.72, last agent dead at
  107.3s. Confirmed by eye: no freezing this time.

## Strategy B: scatter (branch strategy-b)

Ring formation clusters agents into a ring. Turns out that is actively
dangerous: a predator kills every agent touching it in the same tick, no
cooldown, no per-kill limit (environment.py ~line 720), so a tight cluster
that gets caught can lose several agents at once. Strategy B does the
opposite: agents spread out instead of clustering, with a real per-tick
priority order: evade a predator (hard override) > reproduce at 85% of
max_energy > go to the closest visible fruit > wander. A base layer nudges
every agent's path away from others it can see, just enough to separate
paths, not a full reversal.

Reused the wall/obstacle-avoidance code verbatim from the ring-formation
policy (path-based collision check, deflection memory, stall-escape
retreat), since it is generic and already proven to stop agents freezing
near walls.

Evasion detail: while fleeing, the agent turns to face the predator rather
than away from it. Reading predator.py: a predator only breaks into a
slower flanking maneuver instead of a direct sprint chase when the agent
it is chasing is looking towards it and is not too close, so keeping it in
view while retreating degrades its pursuit path.

Wander does not use `rng` at all, since `agent_server.py` recreates it
fresh (same seed) on every single request, so a draw from it does not
actually vary tick to tick when this runs for real. Instead each agent has
a deterministic phase that advances every tick, plus a per-agent starting
offset (golden-angle spaced by agent_id) so agents wander differently from
tick one.

- Strategy B v1: scores on 9 local runs: 544.9, 615.2, 504.8, 904.2, 743.9,
  532.3, 805.4, 708.7, 903.3 (average ~707). Roughly 5x the ring-formation
  policy's ~144 average, and ~33x the original 21.16 baseline. Not yet
  watched live to confirm the intended behavior (scattering, facing down
  predators, no freezing) matches what is actually happening on screen.
- Watched v1 live (seed 2937598241): scored 370.80, population grew as high
  as 21 agents. Spotted a real bug: many agents spinning in place instead
  of moving. Cause: `_wander` computed a full target heading and set
  `turn_angle` to it every tick, but `turn_angle` applies relative to the
  agent's CURRENT facing, not absolute. For most agent_ids that target
  heading is large (near a half turn), so the agent flipped its facing by
  roughly that amount every single tick instead of settling into a
  direction, looking like it was spinning (because it was).
- Strategy B v2: rewrote wander (`_wander_turn`) to turn by a small amount
  each tick (`WANDER_TURN_STEP`, capped well under a full turn) and
  otherwise just walk forward, instead of re-aiming at a big angle every
  tick. Verified with the ground-truth tracer on a 649s / 20+ agent run:
  no sustained stuck streaks anywhere. Scores on 6 fresh runs after the
  fix: 696.2, 645.3, 616.1, 835.6, 729.3, 531.0 (average ~675) — no
  regression from v1's ~707, and the spinning is gone.
- Watched v2 live (seed 1906357515): scored 1271.72, best result so far by
  a wide margin, last agent dead at t=1209.1s out of 3000. Confirmed by
  eye: spinning bug is gone.

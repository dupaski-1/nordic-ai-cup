# Nordic AI Cup 2026: Survival Simulator

My submission for the survival-simulator challenge in the [Nordic AI Cup
2026](https://nordicaicup.com), organised by [Ambolt AI](https://ambolt.io/)
across Sweden, Denmark, Finland, Norway and Iceland.

**The simulator, environment, and challenge rules in this repository are
Ambolt AI's** (see `README.md` and `README_general.md` for their original
docs). The agent policy code is my own work, written as a learning exercise
during a master's in data science.

## The challenge

You control an entire species of herbivores as a hivemind. Every 0.1 simulated
seconds you get each living agent's status and observations (a vision cone
plus a hearing/smell radius) and must return one action per agent: move, turn,
and optionally spawn a new agent. Energy is the core constraint: moving,
turning and spawning all cost energy, sprinting costs far more per unit
distance than walking, and predators spawn over time and kill any agent they
touch. The simulation runs up to 3000 simulated seconds, and score is mainly
how long the species survives, plus a small bonus for eating fruit and a
penalty for agents killed by predators.

The shipped dummy policy scores 21.16, with every agent dead after 21 seconds.

## My strategy

I tried two approaches:

1. **Ring formation** – cluster agents together, seek fruit, avoid walls, and
   spawn once energy is high enough. This worked but revealed a serious flaw:
   predators kill every agent they touch in a single tick with no limit, so a
   tight cluster caught by a predator can lose several agents at once.

2. **Scatter (the one I kept)** – agents spread out instead of clustering,
   with a strict per-tick priority order:
   - Evade a predator (hard override on everything else)
   - Reproduce once energy is high enough
   - Move toward the closest visible fruit
   - Otherwise wander, nudged away from other visible agents to keep paths
     separated

   Two details mattered a lot in practice. First, while fleeing, an agent
   keeps facing the predator rather than turning away, because the predator's
   own pursuit logic switches to a slower flanking maneuver when it knows it
   has been seen. Second, wandering turns by a small fixed step each tick
   instead of aiming at a target heading directly, since large corrections
   are relative to the agent's current facing and otherwise cause it to spin
   in place.

Both strategies share the same wall/obstacle avoidance logic: a path-based
check that samples points along a prospective move and redirects around
anything too close, with a fallback that makes an agent retreat straight away
from nearby obstacles if it gets stuck maneuvering for too long.

## Result

Best confirmed live score: **1271.72**, with the last agent surviving to
t=1209s out of 3000s and the population growing past 20 agents at points
during the run. That's roughly 60x the 21.16 baseline.

## What I'd try next

- Make fruit-seeking and reproduction biome-aware, since biomes affect energy
  cost and fruit spawn rate and the current policy treats them the same.
- Give agents some memory of recent predator sightings instead of reacting
  only to what's currently visible, to react faster to a predator that just
  left the vision cone.
- Tune the reproduction energy threshold with actual data instead of a guess,
  since it trades off population growth against per-agent energy buffer.
- Look into a simple learned policy (e.g. a small neural net trained via
  evolution strategies on the mutating traits already in the game) now that
  the hand-written baseline is solid, since I have no ML background yet and
  this project is also where I'm learning it.

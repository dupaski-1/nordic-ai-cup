import random
import numpy as np
from src.utils.DTOs import ActionRequest

# Scatter strategy.
#
# Agents do not cluster: a predator kills every agent touching it in the
# same tick (no cooldown, no per-kill limit, see environment.py), so a
# tight group that gets caught can lose several agents at once. Priority
# order per tick: evade a predator if one is in sight (hard override, not
# a cost/benefit weighing, since a caught agent just refuels the predator
# to hunt the next one) > reproduce once energy is high enough > head for
# the closest visible fruit > wander if nothing is detected. Underneath
# all of that, an agent nudges its path away from other agents it can see,
# just enough to stop paths from overlapping, not a full reversal.
#
# While evading, the agent turns to face the predator instead of away from
# it. Reading predator.py: a predator only breaks into a slow flanking
# maneuver instead of a direct sprint chase when the agent it is chasing is
# looking towards it and not too close, so keeping the predator in view
# while retreating degrades its pursuit into a less direct path.
#
# rng is not used for anything that needs to vary over time: agent_server.py
# recreates it fresh (same seed) on every single request, so a draw from it
# does not actually change from tick to tick when this runs for real. Wander
# instead uses a per-agent phase that advances every tick.

REPRODUCE_ENERGY_FRACTION = 0.85  # spawn once energy is at least this fraction of max_energy
DIVERGE_RADIUS = 60.0             # start nudging away from another agent within this range
DIVERGE_STRENGTH = 0.5            # how much that nudge can bend the path (kept well under a full reversal)
WANDER_STEP = 0.04                # phase advance per tick while wandering
WANDER_TURN_STEP = 0.15           # max radians turned per tick while wandering (small: see _wander_turn)
WALL_AVOID_DISTANCE = 40.0        # only worry about walls/obstacles closer than this
WALL_BLOCK_CONE = np.pi / 6       # how narrow a direction has to be to count as "aimed at" a wall, for facing only
WALL_CLEARANCE = 6.0              # how much space a move must keep from any wall/obstacle edge (agent size is 5)
STALL_ESCAPE_TICKS = 20           # after this many ticks fighting the same obstacles, stop maneuvering and retreat

_wander_phase = {}  # agent_id -> current phase of its wander heading
_avoid_bias = {}    # agent_id -> +1/-1, which turn direction it's currently deflecting with
_stall_ticks = {}   # agent_id -> consecutive ticks spent actively deflecting around a wall/obstacle


def action_decision(observation_response: dict, rng: random.Random) -> ActionRequest:
    """
    Scatter controller for one agent's turn.

    Args:
        observation_response (dict): Observation response from the environment
        rng (random.Random): Random number generator

    Returns:
        ActionRequest: Action decision
    """
    agent_id = observation_response["agent_id"]
    observations = observation_response["observations"]
    speed = observation_response["speed"]
    sprint_speed = observation_response["sprint_speed"]
    energy = observation_response["energy"]
    max_energy = observation_response["max_energy"]

    predators = [o for o in observations if o.get("type") == "Predator"]
    fruits = [o for o in observations if o.get("type") == "Fruit"]
    nearby_agents = [o for o in observations if o.get("type") == "Agent"]

    evading = bool(predators)

    if evading:
        # Hard override: always flee the closest predator, no weighing it
        # against anything else.
        threat = min(predators, key=lambda o: o["distance"])
        move_x, move_y = _away_from(threat, sprint_speed)
        turn_angle = threat["angle"]  # face it, to force the slower flanking pursuit
    elif fruits:
        target = min(fruits, key=lambda o: o["distance"])
        move_x, move_y = _toward(target, speed)
        turn_angle = np.arctan2(move_y, move_x)
    else:
        move_x, move_y = speed, 0.0  # keep walking in whatever direction we're already facing
        turn_angle = _wander_turn(agent_id)

    if not evading:
        # Fleeing takes priority over politely stepping aside.
        move_x, move_y = _apply_divergence(move_x, move_y, nearby_agents)

    move_x, move_y = _avoid_walls(move_x, move_y, observations, agent_id)
    if not evading:
        # Keeping eyes on the predator matters more than a nicer gaze angle,
        # and this would otherwise fight the intentional "face it" turn.
        turn_angle = _avoid_wall_facing(turn_angle, observations)

    move_distance = np.hypot(move_x, move_y)
    move_direction = np.arctan2(move_y, move_x) if move_distance > 0.1 else 0.0

    spawn_agent = (
        not evading
        and energy > 100  # the game's own spawn floor (environment.py)
        and energy >= REPRODUCE_ENERGY_FRACTION * max_energy
    )

    return ActionRequest(
        agent_id=agent_id,
        move_distance=float(move_distance),
        move_direction=float(move_direction),
        turn_angle=float(turn_angle),
        spawn_agent=spawn_agent,
    )


def _away_from(target, magnitude):
    """Unit vector away from a relative-angle sighting, scaled to magnitude."""
    angle = _wrap(target["angle"] + np.pi)
    return magnitude * np.cos(angle), magnitude * np.sin(angle)


def _toward(target, speed):
    """Move toward a relative-angle sighting, capped at speed."""
    move_x = target["distance"] * np.cos(target["angle"])
    move_y = target["distance"] * np.sin(target["angle"])
    mag = np.hypot(move_x, move_y)
    if mag > speed:
        scale = speed / mag
        move_x *= scale
        move_y *= scale
    return move_x, move_y


def _wander_turn(agent_id):
    """
    A small, slowly-oscillating turn, unique per agent and independent of
    rng (see module docstring). turn_angle applies relative to the agent's
    CURRENT facing, so this has to stay small: turning by anywhere close to
    a full target heading every tick flips "forward" before any real
    distance accumulates, which is what caused agents to visibly spin in
    place instead of travel (an earlier version of this function made
    exactly that mistake). Each agent starts at a different phase
    (golden-angle spacing by agent_id) so they don't all curve the same way
    at the same time.
    """
    start_phase = (agent_id * 2.399963) % (2 * np.pi)
    phase = _wander_phase.get(agent_id, start_phase) + WANDER_STEP
    _wander_phase[agent_id] = phase
    return WANDER_TURN_STEP * np.sin(phase)


def _apply_divergence(move_x, move_y, nearby_agents):
    """
    Nudge (move_x, move_y) away from other agents within DIVERGE_RADIUS,
    stronger the closer they are, then rescale back to the original travel
    distance so this only ever changes direction, never how far the agent
    tries to go. DIVERGE_STRENGTH caps how much the nudge can bend the
    path, so this is always "just enough to separate paths", never a full
    reversal.
    """
    close = [o for o in nearby_agents if o["distance"] < DIVERGE_RADIUS]
    if not close:
        return move_x, move_y

    push_x = push_y = 0.0
    for o in close:
        weight = (DIVERGE_RADIUS - o["distance"]) / DIVERGE_RADIUS
        push_x -= weight * np.cos(o["angle"])
        push_y -= weight * np.sin(o["angle"])

    mag = np.hypot(move_x, move_y)
    if mag < 1e-6:
        return move_x, move_y

    nudged_x = move_x + DIVERGE_STRENGTH * push_x * mag
    nudged_y = move_y + DIVERGE_STRENGTH * push_y * mag
    nudged_mag = np.hypot(nudged_x, nudged_y)
    if nudged_mag < 1e-6:
        return move_x, move_y

    scale = mag / nudged_mag
    return nudged_x * scale, nudged_y * scale


def _edge_midpoints(observations):
    return [
        ((sx + ex) / 2.0, (sy + ey) / 2.0)
        for o in observations if o.get("type") == "Edge"
        for (sx, sy), (ex, ey) in [o["coords"]]
    ]


def _point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from point (px, py) to the closest point on segment (ax, ay)-(bx, by)."""
    abx, aby = bx - ax, by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq < 1e-9:  # the "segment" is really just a point
        return np.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / ab_len_sq
    t = max(0.0, min(1.0, t))  # clamp to the segment, not the infinite line
    cx, cy = ax + t * abx, ay + t * aby
    return np.hypot(px - cx, py - cy)


def _edge_segments(observations):
    return [(sx, sy, ex, ey) for o in observations if o.get("type") == "Edge" for (sx, sy), (ex, ey) in [o["coords"]]]


def _nearby_segments(observations):
    """Only the wall/obstacle edges close enough to matter right now."""
    return [seg for seg in _edge_segments(observations) if _point_segment_distance(0.0, 0.0, *seg) < WALL_AVOID_DISTANCE]


def _path_clear(move_mag, angle, segments):
    """
    Check a handful of points spaced along the straight-line path this
    move would take (not just where it lands) against every nearby wall
    or obstacle edge. Checking the whole path, not just the endpoint,
    catches a move that clips a wall partway through, which matters in
    a tight spot like a gap between the map boundary and an obstacle.
    """
    end_x, end_y = move_mag * np.cos(angle), move_mag * np.sin(angle)
    for t in (0.25, 0.5, 0.75, 1.0):
        px, py = end_x * t, end_y * t
        for seg in segments:
            if _point_segment_distance(px, py, *seg) < WALL_CLEARANCE:
                return False
    return True


def _find_clear_move(base_angle, move_mag, segments, agent_id=None):
    """
    Sweep outward from base_angle in small alternating steps (a bit one
    way, a bit the other, then further out each time) until finding an
    angle whose path stays clear of every nearby wall/obstacle, or give
    up after a full circle (boxed in on all sides, at least one step at
    a time). Only called once we already know base_angle itself isn't
    clear.

    Which way "a bit one way" tries first is remembered per agent
    (_avoid_bias): once an agent starts deflecting left (say) around an
    obstacle, it keeps preferring left on later ticks instead of the
    sweep re-deciding fresh each time. Without that memory, a slightly
    different position each tick can flip which side looks clear first,
    and the agent zigzags left-right-left in place near a tight cluster
    of obstacles instead of committing to one way around, the classic
    "wall following" trick for exactly this.
    """
    bias = _avoid_bias.get(agent_id) if agent_id is not None else None
    signs = (bias, -bias) if bias else (1, -1)

    step = np.pi / 6
    for i in range(1, int(np.pi / step) + 1):
        for sign in signs:
            candidate = _wrap(base_angle + sign * step * i)
            if _path_clear(move_mag, candidate, segments):
                if agent_id is not None:
                    _avoid_bias[agent_id] = sign
                return candidate

    return None  # every direction we tried still clips a wall


def _retreat_from(segments, move_mag):
    """
    Move straight away from the combined center of every nearby wall or
    obstacle edge, ignoring wherever we were actually trying to go.
    Used only once we've been deflecting around the same obstacles for
    a while with no real escape (STALL_ESCAPE_TICKS): heading away from
    all of them at once is close to guaranteed to clear a tight pocket
    between several obstacles, even if it's not the shortest way out,
    which a one-step-at-a-time deflection can't reliably find.
    """
    cx = sum((sx + ex) / 2.0 for sx, sy, ex, ey in segments) / len(segments)
    cy = sum((sy + ey) / 2.0 for sx, sy, ex, ey in segments) / len(segments)
    away_angle = np.arctan2(-cy, -cx)
    return move_mag * np.cos(away_angle), move_mag * np.sin(away_angle)


def _avoid_walls(move_x, move_y, observations, agent_id=None):
    """
    If the intended move's path would come too close to a wall or
    obstacle, redirect it to the nearest clear angle instead (see
    _find_clear_move), keeping the same move distance. If truly nothing
    is clear, hold still rather than pay for a move that just gets
    blocked anyway. If we've had to deflect for too many ticks in a row
    (STALL_ESCAPE_TICKS), stop trying to be clever and retreat instead
    (see _retreat_from).
    """
    segments = _nearby_segments(observations)
    if not segments:
        if agent_id is not None:
            _stall_ticks[agent_id] = 0
        return move_x, move_y

    move_mag = np.hypot(move_x, move_y)
    if move_mag < 0.1:
        return move_x, move_y

    base_angle = np.arctan2(move_y, move_x)

    if agent_id is not None and _stall_ticks.get(agent_id, 0) >= STALL_ESCAPE_TICKS:
        _stall_ticks[agent_id] = 0  # give the retreat a fresh start
        return _retreat_from(segments, move_mag)

    if _path_clear(move_mag, base_angle, segments):
        if agent_id is not None:
            _stall_ticks[agent_id] = 0
            _avoid_bias[agent_id] = None
        return move_x, move_y

    if agent_id is not None:
        _stall_ticks[agent_id] = _stall_ticks.get(agent_id, 0) + 1

    clear_angle = _find_clear_move(base_angle, move_mag, segments, agent_id)
    if clear_angle is None:
        return 0.0, 0.0
    return move_mag * np.cos(clear_angle), move_mag * np.sin(clear_angle)


def _avoid_wall_facing(turn_angle, observations):
    """
    If the facing we're about to turn to would stare straight into a close
    wall, rotate to the nearest angle that isn't aimed straight at a
    nearby wall, toward ground something could actually approach through.
    This is just a gaze heuristic (a narrow cone per wall, not real
    collision geometry), since facing has no physical move to check a
    path for.
    """
    nearby = [(wx, wy) for wx, wy in _edge_midpoints(observations) if np.hypot(wx, wy) < WALL_AVOID_DISTANCE]

    def facing_a_wall(angle):
        return any(abs(_wrap(np.arctan2(wy, wx) - angle)) < WALL_BLOCK_CONE for wx, wy in nearby)

    if not facing_a_wall(turn_angle):
        return turn_angle

    step = np.pi / 6
    for i in range(1, int(np.pi / step) + 1):
        for turn in (step * i, -step * i):
            candidate = _wrap(turn_angle + turn)
            if not facing_a_wall(candidate):
                return candidate

    return turn_angle  # every direction we tried is still aimed at a wall


def _wrap(angle):
    """Wrap an angle to (-pi, pi]."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi

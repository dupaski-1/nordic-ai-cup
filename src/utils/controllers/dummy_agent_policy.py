import random
import numpy as np
from src.utils.DTOs import ActionRequest

# Ring formation policy.
#
# Every agent starts out as its own one-member cluster. Clusters merge
# together when one agent gets close enough to another cluster with room
# to spare, capped at 6 (exact 360 degree coverage at the default vision
# cone). Any agent that spots a fruit heads for it, clustered or not, since
# eating is free once close enough. If nothing's visible, a clustered
# agent's strongest member scouts outward instead of waiting. Clustered
# agents also sweep their facing to cover their share of 360 degrees, since
# below 6 members a fixed cone alone can't. A newborn spawns right next to
# its parent, well within joining range, so it merges into the same
# cluster almost immediately. Predator response is still a later pass.

RING_RADIUS = 18.0        # target distance from the cluster's center, once settled
MIN_SPACING = 14.0        # personal space: push apart from a clustermate closer than this
JOIN_RANGE = 50.0         # within hearing radius, so we're sure to be seen back
SEARCH_TURN = 0.1         # radians per tick, gentle sweep while searching
MAX_CLUSTER_SIZE = 6      # exact 360 coverage at the default 60 degree cone
SPAWN_ENERGY = 100.0      # matches the game's own spawn threshold (environment.py)
SWEEP_STEP = 0.05         # radians per tick added to the coverage-sweep phase
SCOUT_JITTER = 0.5        # radians of randomness in the scout's heading each tick
WALL_AVOID_DISTANCE = 40.0  # only worry about walls/obstacles closer than this
WALL_BLOCK_CONE = np.pi / 6  # how narrow a direction has to be to count as "aimed at" a wall, for facing only
WALL_CLEARANCE = 6.0         # how much space a move must keep from any wall/obstacle edge (agent size is 5)
STALL_ESCAPE_TICKS = 20      # after this many ticks fighting the same obstacles, stop maneuvering and retreat

_cluster_of = {}          # agent_id -> cluster_id
_cluster_members = {}     # cluster_id -> set of agent_id
_last_known_energy = {}   # agent_id -> energy, as of the last time we processed it
_sweep_phase = {}         # agent_id -> current phase of its coverage sweep
_avoid_bias = {}          # agent_id -> +1/-1, which turn direction it's currently deflecting with
_stall_ticks = {}         # agent_id -> consecutive ticks spent actively deflecting around a wall/obstacle


def _ensure_registered(agent_id):
    """The first time we see an agent, it starts as its own one-member cluster."""
    if agent_id not in _cluster_of:
        _cluster_of[agent_id] = agent_id
        _cluster_members[agent_id] = {agent_id}


def _merge_into(agent_id, target_cluster_id):
    """Move agent_id's whole cluster into target_cluster_id's cluster."""
    my_cluster_id = _cluster_of[agent_id]
    if my_cluster_id == target_cluster_id:
        return
    moving = _cluster_members.pop(my_cluster_id, {agent_id})
    for member in moving:
        _cluster_of[member] = target_cluster_id
    _cluster_members[target_cluster_id].update(moving)


def _is_strongest(agent_id, cluster_id):
    """True if no other member of the cluster has a higher last-known energy."""
    my_energy = _last_known_energy.get(agent_id, float("-inf"))
    for member_id in _cluster_members[cluster_id]:
        if member_id != agent_id and _last_known_energy.get(member_id, float("-inf")) > my_energy:
            return False
    return True


def action_decision(observation_response: dict, rng: random.Random) -> ActionRequest:
    """
    Ring formation controller for one agent's turn.

    Args:
        observation_response (dict): Observation response from the environment
        rng (random.Random): Random number generator

    Returns:
        ActionRequest: Action decision
    """
    agent_id = observation_response["agent_id"]
    observations = observation_response["observations"]
    speed = observation_response["speed"]
    energy = observation_response["energy"]
    vision_angle = observation_response["vision_angle"]

    _last_known_energy[agent_id] = energy
    _ensure_registered(agent_id)
    my_cluster_id = _cluster_of[agent_id]

    clustermates = [
        o for o in observations
        if o.get("type") == "Agent" and _cluster_of.get(o.get("id")) == my_cluster_id
    ]

    predator_in_sight = any(o.get("type") == "Predator" for o in observations)
    fruits = [] if predator_in_sight else [o for o in observations if o.get("type") == "Fruit"]

    if fruits:
        # Any agent heads for a fruit it spots, clustered or not, this
        # does not depend on currently seeing a clustermate too.
        target = min(fruits, key=lambda o: o["distance"])
        move_distance, move_direction = _move_toward(target, speed, observations, agent_id)
        if clustermates:
            turn_angle = _facing(clustermates, agent_id, my_cluster_id, vision_angle, observations)
        else:
            turn_angle = target["angle"]

    elif clustermates:
        move_distance, move_direction, turn_angle = _settle(
            clustermates, speed, agent_id=agent_id, cluster_id=my_cluster_id,
            observations=observations, vision_angle=vision_angle, rng=rng,
        )

    else:
        joinable = []
        for o in observations:
            if o.get("type") != "Agent":
                continue
            other_cluster_id = _cluster_of.get(o.get("id"))
            if other_cluster_id is None:
                continue  # that agent hasn't registered itself yet, try again next tick
            if len(_cluster_members[other_cluster_id]) < MAX_CLUSTER_SIZE:
                joinable.append(o)

        if joinable:
            closest = min(joinable, key=lambda o: o["distance"])
            if closest["distance"] <= JOIN_RANGE:
                _merge_into(agent_id, _cluster_of[closest["id"]])
                move_distance, move_direction, turn_angle = _settle(
                    [closest], speed, agent_id=agent_id, cluster_id=_cluster_of[agent_id],
                    observations=observations, vision_angle=vision_angle, rng=rng,
                )
            else:
                move_distance, move_direction = _move_toward(closest, speed, observations, agent_id)
                turn_angle = closest["angle"]
        else:
            move_x, move_y = _avoid_walls(speed, 0.0, observations, agent_id)
            move_distance = min(speed, np.hypot(move_x, move_y))
            move_direction = np.arctan2(move_y, move_x) if move_distance > 0.1 else 0.0
            turn_angle = _avoid_wall_facing(rng.uniform(-SEARCH_TURN, SEARCH_TURN), observations)

    current_cluster_size = len(_cluster_members[_cluster_of[agent_id]])
    spawn_agent = energy > SPAWN_ENERGY and current_cluster_size < MAX_CLUSTER_SIZE

    return ActionRequest(
        agent_id=agent_id,
        move_distance=move_distance,
        move_direction=move_direction,
        turn_angle=turn_angle,
        spawn_agent=spawn_agent,
    )


def _settle(ring_sightings, speed, agent_id=None, cluster_id=None, observations=None,
            vision_angle=None, rng=None):
    """
    Reactive hold: move toward RING_RADIUS from the visible group's center,
    push apart from a too-close neighbor, and turn to face outward (with a
    slow sweep added below 6 members, since a fixed cone can't cover the
    full 360 alone). If nobody in view is a predator, the cluster's
    strongest (highest last-known energy) member scouts outward with some
    randomness instead of holding, since it can best afford the walking
    cost, everyone else holds. A nearby wall in the way skips that move
    rather than paying for one that just gets blocked anyway.
    """
    if not ring_sightings:
        return 0.0, 0.0, 0.0

    dx = sum(o["distance"] * np.cos(o["angle"]) for o in ring_sightings) / len(ring_sightings)
    dy = sum(o["distance"] * np.sin(o["angle"]) for o in ring_sightings) / len(ring_sightings)
    center_angle = np.arctan2(dy, dx)
    center_dist = np.hypot(dx, dy)

    radial_error = center_dist - RING_RADIUS
    move_x = radial_error * np.cos(center_angle)
    move_y = radial_error * np.sin(center_angle)

    nearest = min(ring_sightings, key=lambda o: o["distance"])
    if nearest["distance"] < MIN_SPACING:
        push = MIN_SPACING - nearest["distance"]
        move_x -= push * np.cos(nearest["angle"])
        move_y -= push * np.sin(nearest["angle"])

    if observations is not None:
        predator_in_sight = any(o.get("type") == "Predator" for o in observations)
        if not predator_in_sight and agent_id is not None and _is_strongest(agent_id, cluster_id):
            scout_angle = rng.uniform(-SCOUT_JITTER, SCOUT_JITTER) if rng else 0.0
            move_x = speed * np.cos(scout_angle)
            move_y = speed * np.sin(scout_angle)
        move_x, move_y = _avoid_walls(move_x, move_y, observations, agent_id)

    sweep_offset = _sweep_offset(agent_id, cluster_id, vision_angle)
    move_distance = min(speed, np.hypot(move_x, move_y))
    move_direction = np.arctan2(move_y, move_x) if move_distance > 0.1 else 0.0
    turn_angle = _wrap(center_angle + np.pi + sweep_offset)
    if observations is not None:
        turn_angle = _avoid_wall_facing(turn_angle, observations)

    return float(move_distance), float(move_direction), float(turn_angle)


def _facing(clustermates, agent_id, cluster_id, vision_angle, observations=None):
    """Relative turn needed to face directly away from the visible group's
    center, plus a coverage sweep below 6 members. Used to keep a clustered
    agent's facing consistent even on ticks where it's off chasing a fruit
    rather than holding position."""
    dx = sum(o["distance"] * np.cos(o["angle"]) for o in clustermates) / len(clustermates)
    dy = sum(o["distance"] * np.sin(o["angle"]) for o in clustermates) / len(clustermates)
    center_angle = np.arctan2(dy, dx)
    sweep_offset = _sweep_offset(agent_id, cluster_id, vision_angle)
    turn_angle = _wrap(center_angle + np.pi + sweep_offset)
    if observations is not None:
        turn_angle = _avoid_wall_facing(turn_angle, observations)
    return turn_angle


def _sweep_offset(agent_id, cluster_id, vision_angle):
    """
    How far off dead-outward to bias facing this tick. Below 6 members, a
    fixed cone can't cover the full 360 degrees alone, so this slowly
    oscillates back and forth to sweep across the share of the circle the
    gap leaves uncovered, sized from the cluster's current size and this
    agent's actual vision cone width. At 6 members the gap is zero and
    facing just stays put.
    """
    if cluster_id is None or vision_angle is None:
        return 0.0
    cluster_size = len(_cluster_members[cluster_id])
    gap = (2 * np.pi / cluster_size) - vision_angle
    if gap <= 0:
        return 0.0
    phase = _sweep_phase.get(agent_id, 0.0) + SWEEP_STEP
    _sweep_phase[agent_id] = phase
    return (gap / 2.0) * np.sin(phase)


def _move_toward(target, speed, observations, agent_id=None):
    """Move toward a relative-angle sighting (a fruit or another agent),
    skipping the move instead of paying for one that a nearby wall blocks."""
    move_x = target["distance"] * np.cos(target["angle"])
    move_y = target["distance"] * np.sin(target["angle"])
    move_x, move_y = _avoid_walls(move_x, move_y, observations, agent_id)
    move_distance = min(speed, np.hypot(move_x, move_y))
    move_direction = np.arctan2(move_y, move_x) if move_distance > 0.1 else 0.0
    return float(move_distance), float(move_direction)


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
    wall (as happens at a map corner, where "face away from the cluster"
    can point past the boundary), rotate to the nearest angle that isn't
    aimed straight at a nearby wall, toward ground something could
    actually approach through. This is just a gaze heuristic (a narrow
    cone per wall, not real collision geometry), since facing has no
    physical move to check a path for.
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

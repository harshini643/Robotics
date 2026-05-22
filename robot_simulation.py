import pybullet as p
import pybullet_data
import random
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Initialize PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load environment
ground = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
road = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, globalScaling=10)
p.changeVisualShape(road, -1, rgbaColor=[0.5, 0.5, 0.5, 1])

# Delivery Bases
bases = {
    "base1": (5, -5, 0.1),   # Green
    "base2": (-5, 5, 0.1),   # Red
    "base3": (5, 5, 0.1),    # Blue
    "base4": (-5, -5, 0.1)   # Yellow
}

# Starting positions for robots
robot_start_positions = {
    "robot1": (6, -6, 0.1),   # Green robot
    "robot2": (-6, 6, 0.1),   # Red robot
    "robot3": (6, 6, 0.1),    # Blue robot
    "robot4": (-6, -6, 0.1)   # Yellow robot
}

# Base colors
base_colors = {
    "base1": [0.2, 0.7, 0.2, 0.7],  # Green
    "base2": [0.7, 0.2, 0.2, 0.7],  # Red
    "base3": [0.2, 0.2, 0.7, 0.7],  # Blue
    "base4": [0.7, 0.7, 0.2, 0.7]   # Yellow
}

# Robot colors
robot_colors = {
    "robot1": [0.2, 0.7, 0.2, 1.0],  # Green
    "robot2": [0.7, 0.2, 0.2, 1.0],  # Red
    "robot3": [0.2, 0.2, 0.7, 1.0],  # Blue
    "robot4": [0.7, 0.7, 0.2, 1.0]   # Yellow
}

# Initialize packages
packages = {
    "base1": 3,
    "base2": 3,
    "base3": 3,
    "base4": 3
}

# Physical package boxes
package_objects = {
    "base1": [],
    "base2": [],
    "base3": [],
    "base4": []
}

# Robot constraints
robot_constraints = {}

# Communication Hub
communication_hub = {
    "messages": [],
    "robot_status": {}
}

# Track communication hub activity for graphing
communication_hub_activity = []  # [(timestamp, num_messages), ...]

# Cooldown for communication hub status print
last_status_print = 0
STATUS_PRINT_COOLDOWN = 5

# Track message boxes in the simulation
message_boxes = {}

# Track dialog cooldowns
dialog_cooldowns = {}

# Simulation results
simulation_results = {
    "travel_times": {},
    "intermediate_stops": {},
    "start_times": {}
}

# Load Robots
robot1 = p.loadURDF("r2d2.urdf", robot_start_positions["robot1"])
robot2 = p.loadURDF("r2d2.urdf", robot_start_positions["robot2"])
robot3 = p.loadURDF("r2d2.urdf", robot_start_positions["robot3"])
robot4 = p.loadURDF("r2d2.urdf", robot_start_positions["robot4"])

# Change robot colors
p.changeVisualShape(robot1, -1, rgbaColor=robot_colors["robot1"])
p.changeVisualShape(robot2, -1, rgbaColor=robot_colors["robot2"])
p.changeVisualShape(robot3, -1, rgbaColor=robot_colors["robot3"])
p.changeVisualShape(robot4, -1, rgbaColor=robot_colors["robot4"])

# Robot configuration
ROBOT_STEP_SIZE = 0.1  # Increased for faster movement
DETECTION_RADIUS = 5.5
COLLISION_RADIUS = 1.0
PASSING_RADIUS = 1.3
AVOIDANCE_STRENGTH = 8.0
ROBOT_AVOIDANCE_WEIGHT = 0.45
CAR_AVOIDANCE_WEIGHT = 1.2
MIN_DIRECTION_CHANGE = 0.15  # Balanced value for better pathing
MAX_STUCK_COUNT = 50
ROTATION_SMOOTHING = 0.1
SAFETY_BUFFER = 0.8
DELIVERY_THRESHOLD = 0.5
SIMULATION_DELAY = 0.02
ROBOT_SPACING = 0.5
BASE_RADIUS = 2.0
MESSAGE_BOX_LIFETIME = 3.0
DIALOG_HEIGHT_OFFSET = 0.2
DIALOG_COOLDOWN = 3.0

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def normalize_vector(vector):
    norm = math.sqrt(vector[0]**2 + vector[1]**2)
    if norm < 0.01:
        return (0, 0)
    return (vector[0]/norm, vector[1]/norm)

def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def send_message(robot_id, message):
    state = robot_states[robot_id]
    timestamp = time.time()
    communication_hub["messages"].append({
        "sender": state["name"],
        "color": state["color"],
        "message": message,
        "timestamp": timestamp
    })
    communication_hub["messages"] = [msg for msg in communication_hub["messages"] 
                                   if timestamp - msg["timestamp"] < 10]

def update_robot_status(robot_id):
    state = robot_states[robot_id]
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    communication_hub["robot_status"][state["name"]] = {
        "position": pos,
        "target_base": state["target_base"],
        "delivered": state["delivered"],
        "carrying_package": bool(state["carrying_package"]),
        "timestamp": time.time()
    }

def create_package(base_name):
    base_pos = bases[base_name]
    offset_x = random.uniform(-0.3, 0.3)
    offset_y = random.uniform(-0.3, 0.3)
    package = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05], 
                                 rgbaColor=[0.8, 0.5, 0.2, 1.0])
    package_id = p.createMultiBody(
        baseMass=0.1,
        baseVisualShapeIndex=package,
        basePosition=[base_pos[0] + offset_x, base_pos[1] + offset_y, base_pos[2] + 0.05]
    )
    return package_id

# Robot states
robot_states = {
    robot1: {
        "name": "robot1",
        "color": "Green",
        "current_base": "base1",
        "start_base": "base1",
        "target_base": None,
        "delivered": True,
        "avoiding": False,
        "carrying_package": None,
        "visible_package": None,
        "stuck_counter": 0,
        "last_pos": robot_start_positions["robot1"],
        "last_orientation": p.getQuaternionFromEuler([0, 0, 0]),
        "delivery_count": 0,
        "path_history": [],
        "has_traveled_message_printed": False,
        "returning": False,
        "last_move_vector": (0, 0),
        "pause_time": None,
        "pause_base": None,
        "is_paused": False,
        "stop_at_intermediate": False,
        "intermediate_stop_bases": [],
        "intermediate_stops": 0,
        "stopped_bases": set(),
        "returned_message_printed": False,
        "current_intermediate_index": 0,
        "avoidance_count": 0,
        "distance_history": [],
        "event_timeline": [],
        "speed_factor": random.uniform(0.9, 1.15),
        "stop_duration": random.uniform(3, 8),
        "completion_time":None
    },
    robot2: {
        "name": "robot2",
        "color": "Red",
        "current_base": "base2",
        "start_base": "base2",
        "target_base": None,
        "delivered": True,
        "avoiding": False,
        "carrying_package": None,
        "visible_package": None,
        "stuck_counter": 0,
        "last_pos": robot_start_positions["robot2"],
        "last_orientation": p.getQuaternionFromEuler([0, 0, 0]),
        "delivery_count": 0,
        "path_history": [],
        "has_traveled_message_printed": False,
        "returning": False,
        "last_move_vector": (0, 0),
        "pause_time": None,
        "pause_base": None,
        "is_paused": False,
        "stop_at_intermediate": False,
        "intermediate_stop_bases": [],
        "intermediate_stops": 0,
        "stopped_bases": set(),
        "returned_message_printed": False,
        "current_intermediate_index": 0,
        "avoidance_count": 0,
        "distance_history": [],
        "event_timeline": [],
        "speed_factor": random.uniform(0.9, 1.15),
        "stop_duration": random.uniform(3, 8),
        "completion_time":None
    },
    robot3: {
        "name": "robot3",
        "color": "Blue",
        "current_base": "base3",
        "start_base": "base3",
        "target_base": None,
        "delivered": True,
        "avoiding": False,
        "carrying_package": None,
        "visible_package": None,
        "stuck_counter": 0,
        "last_pos": robot_start_positions["robot3"],
        "last_orientation": p.getQuaternionFromEuler([0, 0, 0]),
        "delivery_count": 0,
        "path_history": [],
        "has_traveled_message_printed": False,
        "returning": False,
        "last_move_vector": (0, 0),
        "pause_time": None,
        "pause_base": None,
        "is_paused": False,
        "stop_at_intermediate": False,
        "intermediate_stop_bases": [],
        "intermediate_stops": 0,
        "stopped_bases": set(),
        "returned_message_printed": False,
        "current_intermediate_index": 0,
        "avoidance_count": 0,
        "distance_history": [],
        "event_timeline": [],
        "speed_factor": random.uniform(0.9, 1.15),
        "stop_duration": random.uniform(3, 8),
        "completion_time":None
    },
    robot4: {
        "name": "robot4",
        "color": "Yellow",
        "current_base": "base4",
        "start_base": "base4",
        "target_base": None,
        "delivered": True,
        "avoiding": False,
        "carrying_package": None,
        "visible_package": None,
        "stuck_counter": 0,
        "last_pos": robot_start_positions["robot4"],
        "last_orientation": p.getQuaternionFromEuler([0, 0, 0]),
        "delivery_count": 0,
        "path_history": [],
        "has_traveled_message_printed": False,
        "returning": False,
        "last_move_vector": (0, 0),
        "pause_time": None,
        "pause_base": None,
        "is_paused": False,
        "stop_at_intermediate": False,
        "intermediate_stop_bases": [],
        "intermediate_stops": 0,
        "stopped_bases": set(),
        "returned_message_printed": False,
        "current_intermediate_index": 0,
        "avoidance_count": 0,
        "distance_history": [],
        "event_timeline": [],
        "speed_factor": random.uniform(0.9, 1.15),
        "stop_duration": random.uniform(3, 8),
        "completion_time":None

    }
}

base_name_colors = {
    "base1": "Green",
    "base2": "Red",
    "base3": "Blue",
    "base4": "Yellow"
}

color_to_base = {
    "green": "base1",
    "red": "base2",
    "blue": "base3",
    "yellow": "base4"
}
# =========================
# MOVING CAR SYSTEM
# =========================

cars = []

# Fixed lanes
car_lanes = [-4, -2, 2, 4]

# Store metadata for each car
car_data = {}

ROAD_LIMIT = 12


def setup_random_cars(num_cars=6):
    """
    Creates smooth moving cars with proper velocities
    """
    global cars, car_data

    # Remove old cars
    for c in cars:
        p.removeBody(c)

    cars = []
    car_data = {}

    for i in range(num_cars):

        lane = car_lanes[i % len(car_lanes)]

        # Random starting x
        start_x = random.uniform(-ROAD_LIMIT, ROAD_LIMIT)

        # Alternate directions
        direction = 1 if i % 2 == 0 else -1

        car_scale = random.uniform(0.5, 0.7)

        car = p.loadURDF(
            "racecar/racecar.urdf",
            [start_x, lane, 0.1],
            globalScaling=car_scale
        )

        # Smooth speed
        speed = random.uniform(0.03, 0.06)

        # Orientation based on direction
        yaw = 0 if direction == 1 else math.pi
        orientation = p.getQuaternionFromEuler([0, 0, yaw])

        p.resetBasePositionAndOrientation(
            car,
            [start_x, lane, 0.1],
            orientation
        )

        cars.append(car)

        car_data[car] = {
            "lane": lane,
            "speed": speed,
            "direction": direction,
            "yaw": yaw
        }


setup_random_cars()

def move_cars():

    for car in cars:

        pos, orn = p.getBasePositionAndOrientation(car)

        data = car_data[car]

        speed = data["speed"]
        direction = data["direction"]
        lane = data["lane"]
        yaw = data["yaw"]

        # Smooth forward motion
        new_x = pos[0] + speed * direction

        # Small smooth sine-wave drift
        drift = 0.08 * math.sin(time.time() * 1.5 + car)

        new_y = lane + drift

        # Loop road
        if direction == 1 and new_x > ROAD_LIMIT:
            new_x = -ROAD_LIMIT

        elif direction == -1 and new_x < -ROAD_LIMIT:
            new_x = ROAD_LIMIT

        # Apply position
        p.resetBasePositionAndOrientation(
            car,
            [new_x, new_y, 0.1],
            p.getQuaternionFromEuler([0, 0, yaw])
        )

        # =========================
        # DEBUG LABEL FOR CARS
        # =========================
        p.addUserDebugText(
            "CAR",
            [new_x, new_y, 0.8],
            textColorRGB=[1, 0, 0],
            textSize=1.2,
            lifeTime=0.1
        )

def create_robot_package(robot_id):
    robot_pos, _ = p.getBasePositionAndOrientation(robot_id)
    state = robot_states[robot_id]
    robot_name = state["name"]
    robot_rgba = robot_colors[robot_name]
    package_color = [
        min(0.9, robot_rgba[0] + 0.2),
        min(0.6, robot_rgba[1] + 0.1),
        min(0.1, robot_rgba[2] + 0.05),
        1.0
    ]
    package = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.04], 
                                 rgbaColor=package_color)
    package_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=package,
        basePosition=[robot_pos[0], robot_pos[1], robot_pos[2] + 0.25]
    )
    return package_id

def update_robot_package_position(robot_id, package_id):
    if package_id is not None:
        robot_pos, robot_orn = p.getBasePositionAndOrientation(robot_id)
        p.resetBasePositionAndOrientation(
            package_id,
            [robot_pos[0], robot_pos[1], robot_pos[2] + 0.25],
            robot_orn
        )

def pick_up_package(robot_id):
    state = robot_states[robot_id]
    current_base = state["current_base"]
    if packages[current_base] > 0 and len(package_objects[current_base]) > 0:
        package_id = package_objects[current_base].pop()
        p.removeBody(package_id)
        visible_package = create_robot_package(robot_id)
        state["visible_package"] = visible_package
        state["carrying_package"] = {
            "from_base": current_base,
            "to_base": state["target_base"]
        }
        send_message(robot_id, f"Picked up package from {base_name_colors[current_base]} Base")
        print(f"{state['color']} Robot picked up a package from {base_name_colors[current_base]} Base")
        return True
    else:
        print(f"No packages available at {base_name_colors[current_base]} Base for {state['color']} Robot!")
        state["delivered"] = True
        return False

def deliver_package(robot_id, target_base=None):
    state = robot_states[robot_id]
    if target_base is None:
        target_base = state["target_base"]
    
    if state["carrying_package"]:
        if state["visible_package"] is not None:
            p.removeBody(state["visible_package"])
            state["visible_package"] = None
        
        new_package = create_package(target_base)
        package_objects[target_base].append(new_package)
        
        packages[state["current_base"]] -= 1
        packages[target_base] += 1
        
        state["carrying_package"] = None
        state["delivery_count"] += 1
        
        send_message(robot_id, f"Delivered package to {base_name_colors[target_base]} Base")
        print(f"{state['color']} Robot delivered the package to {base_name_colors[target_base]} Base!, (Delivery #{state['delivery_count']})")
        
        state["current_base"] = target_base
        
        if target_base == state["start_base"]:
            state["returning"] = False
            state["delivered"] = True
            state["target_base"] = None
            state["has_traveled_message_printed"] = False
            state["current_intermediate_index"] = 0
            state["intermediate_stop_bases"] = []
            state["stopped_bases"] = set()
        elif target_base != state["target_base"]:
            # Intermediate delivery, pick up a new package
            state["visible_package"] = create_robot_package(robot_id)
            state["carrying_package"] = {"from_base": target_base, "to_base": state["target_base"]}
            send_message(robot_id, f"Picked up new package at {base_name_colors[target_base]} Base for {base_name_colors[state['target_base']]}")
        
        return True
    return False

def is_collision_free(robot_id, new_pos):
    robot_aabb_min, robot_aabb_max = p.getAABB(robot_id)
    robot_width = max(robot_aabb_max[0] - robot_aabb_min[0], robot_aabb_max[1] - robot_aabb_min[1])
    for car in cars:
        car_pos, _ = p.getBasePositionAndOrientation(car)
        dist = distance(new_pos, car_pos)
        min_safe_dist = (robot_width + robot_width) / 2 * 1.4 + SAFETY_BUFFER
        if dist < min_safe_dist:
            return False
    for other_robot in [robot1, robot2, robot3, robot4]:
        if other_robot != robot_id:
            other_pos, _ = p.getBasePositionAndOrientation(other_robot)
            dist = distance(new_pos, other_pos)
            min_safe_dist = (robot_width + robot_width) / 2 * 1.4 + SAFETY_BUFFER
            if dist < min_safe_dist:
                return False
    return True

def lock_robot(robot_id):
    if robot_id not in robot_constraints:
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        constraint_id = p.createConstraint(
            robot_id, -1,
            -1, -1,
            p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[pos[0], pos[1], pos[2]],
            childFrameOrientation=orn
        )
        robot_constraints[robot_id] = constraint_id

def unlock_robot(robot_id):
    if robot_id in robot_constraints:
        p.removeConstraint(robot_constraints[robot_id])
        del robot_constraints[robot_id]

def detect_obstacles(robot_id):
    robot_pos, _ = p.getBasePositionAndOrientation(robot_id)
    obstacles = []
    robot_aabb_min, robot_aabb_max = p.getAABB(robot_id)
    robot_width = max(robot_aabb_max[0] - robot_aabb_min[0], robot_aabb_max[1] - robot_aabb_min[1])
    
    for car in cars:
        car_pos, _ = p.getBasePositionAndOrientation(car)
        dist = distance(robot_pos, car_pos)
        min_safe_dist = (robot_width + robot_width) / 2 * 1.2 + SAFETY_BUFFER
        if dist < DETECTION_RADIUS:
            direction = (car_pos[0] - robot_pos[0], car_pos[1] - robot_pos[1])
            obstacles.append({
                "pos": car_pos,
                "type": "car",
                "direction": direction,
                "distance": dist,
                "min_safe_dist": min_safe_dist
            })
    
    for other_robot in [robot1, robot2, robot3, robot4]:
        if other_robot != robot_id:
            other_pos, _ = p.getBasePositionAndOrientation(other_robot)
            dist = distance(robot_pos, other_pos)
            min_safe_dist = (robot_width + robot_width) / 2 * 1.2 + SAFETY_BUFFER
            if dist < DETECTION_RADIUS:
                direction = (other_pos[0] - robot_pos[0], other_pos[1] - robot_pos[1])
                obstacles.append({
                    "pos": other_pos,
                    "type": "robot",
                    "direction": direction,
                    "distance": dist,
                    "min_safe_dist": min_safe_dist
                })
    
    return sorted(obstacles, key=lambda x: x["distance"])

def calculate_avoidance_vector(robot_id, obstacles, goal_pos):

    current_pos, _ = p.getBasePositionAndOrientation(robot_id)

    # Goal direction
    goal_vector = (
        goal_pos[0] - current_pos[0],
        goal_pos[1] - current_pos[1]
    )

    goal_vector = normalize_vector(goal_vector)

    avoidance_vector = [0, 0]

    avoiding = False

    for obstacle in obstacles:

        dist = obstacle["distance"]

        if dist > DETECTION_RADIUS:
            continue

        # --------------------------------
        # REPULSION VECTOR
        # --------------------------------
        repel_x = current_pos[0] - obstacle["pos"][0]
        repel_y = current_pos[1] - obstacle["pos"][1]

        repel = normalize_vector((repel_x, repel_y))

        # Stronger if closer
        strength = (DETECTION_RADIUS - dist) / DETECTION_RADIUS

        # --------------------------------
        # DIFFERENT WEIGHTS
        # --------------------------------
        if obstacle["type"] == "robot":
            strength *= ROBOT_AVOIDANCE_WEIGHT
        else:
            strength *= CAR_AVOIDANCE_WEIGHT

        # --------------------------------
        # FIX OSCILLATION:
        # ALWAYS PASS ON SAME SIDE
        # --------------------------------
        side_vector = (-repel[1], repel[0])

        # Deterministic side selection
        if robot_id % 2 == 0:
            side_vector = (-side_vector[0], -side_vector[1])

        avoidance_vector[0] += repel[0] * strength
        avoidance_vector[1] += repel[1] * strength

        avoidance_vector[0] += side_vector[0] * strength * 0.6
        avoidance_vector[1] += side_vector[1] * strength * 0.6

        avoiding = True

    # --------------------------------
    # COMBINE WITH GOAL
    # --------------------------------
    final_vector = [
        goal_vector[0] + avoidance_vector[0],
        goal_vector[1] + avoidance_vector[1]
    ]

    final_vector = normalize_vector(final_vector)

    # --------------------------------
    # SMOOTH MOVEMENT MEMORY
    # --------------------------------
    state = robot_states[robot_id]

    prev = state["last_move_vector"]

    smooth_vector = [
        0.75 * prev[0] + 0.25 * final_vector[0],
        0.75 * prev[1] + 0.25 * final_vector[1]
    ]

    smooth_vector = normalize_vector(smooth_vector)

    return smooth_vector, avoiding

def get_next_intermediate_base(robot_id, current_pos):
    state = robot_states[robot_id]
    if not state["stop_at_intermediate"] or state["returning"]:
        return None, float('inf')
    
    # Check the next intermediate base in the list
    if state["current_intermediate_index"] < len(state["intermediate_stop_bases"]):
        base_name = state["intermediate_stop_bases"][state["current_intermediate_index"]]
        if base_name in bases and base_name not in state["stopped_bases"]:
            base_pos = bases[base_name]
            dist = distance(current_pos, base_pos)
            return base_name, dist
    return None, float('inf')

def get_offset_goal(robot_id, goal_pos):

    offsets = {
        robot1: (0.8, 0.8),
        robot2: (-0.8, 0.8),
        robot3: (0.8, -0.8),
        robot4: (-0.8, -0.8)
    }

    offset = offsets[robot_id]

    return (
        goal_pos[0] + offset[0],
        goal_pos[1] + offset[1],
        goal_pos[2]
    )

def robot_controller(robot_id):

    state = robot_states[robot_id]

    update_robot_status(robot_id)

    # =====================================
    # STOP IF FINISHED
    # =====================================
    if state["delivered"] and not state["returning"]:
        lock_robot(robot_id)
        return

    if state["target_base"] is None:
        lock_robot(robot_id)
        return

    current_pos, current_orn = p.getBasePositionAndOrientation(robot_id)

    # =====================================
    # DEADLOCK DETECTION
    # =====================================
    movement = distance(current_pos, state["last_pos"])

    if movement < 0.01:
        state["stuck_counter"] += 1
    else:
        state["stuck_counter"] = 0

    # Escape if stuck
    if state["stuck_counter"] > 40:

        escape_angle = random.uniform(0, 2 * math.pi)

        escape_vector = (
            math.cos(escape_angle),
            math.sin(escape_angle)
        )

        emergency_pos = [
            current_pos[0] + escape_vector[0] * 1.2,
            current_pos[1] + escape_vector[1] * 1.2,
            current_pos[2]
        ]

        p.resetBasePositionAndOrientation(
            robot_id,
            emergency_pos,
            current_orn
        )

        state["stuck_counter"] = 0

        print(f"{state['color']} Robot escaped deadlock")

    # =====================================
    # DISTANCE TRACKING
    # =====================================
    if state["last_pos"] is not None:

        dist_increment = distance(current_pos, state["last_pos"])

        current_time = time.time()

        if state["distance_history"]:
            last_dist = state["distance_history"][-1][1]

            state["distance_history"].append(
                (current_time, last_dist + dist_increment)
            )

        else:
            state["distance_history"].append(
                (current_time, dist_increment)
            )

    # =====================================
    # PAUSE HANDLING
    # =====================================
    if state["is_paused"]:

        if time.time() - state["pause_time"] < state["stop_duration"]:
            lock_robot(robot_id)
            return

        else:

            print(
                f"{state['color']} Robot is resuming from "
                f"{state['pause_base']} toward "
                f"{base_name_colors[state['target_base']]}"
            )

            send_message(
                robot_id,
                f"Resuming to {base_name_colors[state['target_base']]} Base"
            )

            state["is_paused"] = False
            state["pause_time"] = None
            state["pause_base"] = None

            unlock_robot(robot_id)

    # =====================================
    # RETURNING HOME
    # =====================================
    if state["returning"]:

        goal_pos = get_offset_goal(
            robot_id,
            robot_start_positions[state["name"]]
        )

        dist_to_goal = distance(current_pos, goal_pos)

        if dist_to_goal < BASE_RADIUS:

            deliver_package(robot_id, state["start_base"])

            # Record exact completion time ONCE
            if state["completion_time"] is None:

                state["completion_time"] = time.time()

                simulation_results["travel_times"][state["name"]] = (
                    state["completion_time"]
                    - simulation_results["start_times"][state["name"]]
                )

            simulation_results["intermediate_stops"][state["name"]] = (
                state["intermediate_stops"]
            )

            state["event_timeline"].append(
                ("Returned", time.time())
            )

            if not state["returned_message_printed"]:

                print(
                    f"{state['color']} Robot has returned "
                    f"to its starting position."
                )

                state["returned_message_printed"] = True

            return

    # =====================================
    # GOING TO TARGET
    # =====================================
    else:

        goal_pos = get_offset_goal(
            robot_id,
            bases[state["target_base"]]
        )

        dist_to_goal = distance(current_pos, goal_pos)

        if dist_to_goal < BASE_RADIUS:

            deliver_package(robot_id)

            state["event_timeline"].append(
                ("Delivered", time.time())
            )

            state["returning"] = True

            state["has_traveled_message_printed"] = False

            state["current_intermediate_index"] = 0

            return

    # =====================================
    # INTERMEDIATE STOPS
    # =====================================
    next_base, min_dist = get_next_intermediate_base(
        robot_id,
        current_pos
    )

    if next_base and min_dist < BASE_RADIUS:

        print(
            f"{state['color']} Robot is stopping at "
            f"{base_name_colors[next_base]} Base"
        )

        send_message(
            robot_id,
            f"Stopping at {base_name_colors[next_base]} Base"
        )

        state["is_paused"] = True

        # Random realistic unloading delay
        state["stop_duration"] = random.uniform(4, 10)

        state["pause_time"] = time.time()

        state["pause_base"] = base_name_colors[next_base]

        state["intermediate_stops"] += 1

        state["stopped_bases"].add(next_base)

        state["current_intermediate_index"] += 1

        state["event_timeline"].append(
            (
                f"Stopped at {base_name_colors[next_base]}",
                time.time()
            )
        )

        deliver_package(robot_id, next_base)

        lock_robot(robot_id)

        return

    elif next_base:

        goal_pos = bases[next_base]

    # =====================================
    # OBSTACLE AVOIDANCE
    # =====================================
    obstacles = detect_obstacles(robot_id)

    move_vector, is_avoiding = calculate_avoidance_vector(
        robot_id,
        obstacles,
        goal_pos
    )

    # Count only NEW avoidance events
    if is_avoiding and not state["avoiding"]:
        state["avoidance_count"] += 1

    state["avoiding"] = is_avoiding

    state["last_move_vector"] = move_vector

    # =====================================
    # MOVEMENT
    # =====================================
    if not state["delivered"]:

        if not state["has_traveled_message_printed"]:

            msg = (
                f"Heading to "
                f"{base_name_colors[state['target_base']]} Base"
                if not state["returning"]
                else
                f"Returning to "
                f"{base_name_colors[state['start_base']]} Base"
            )

            send_message(robot_id, msg)

            print(f"{state['color']} Robot: {msg}")

            state["has_traveled_message_printed"] = True

            state["event_timeline"].append(
                ("Started", time.time())
            )

        # Speed variation
        speed = ROBOT_STEP_SIZE * state["speed_factor"]

        new_pos = [
            current_pos[0] + move_vector[0] * speed,
            current_pos[1] + move_vector[1] * speed,
            current_pos[2]
        ]

        target_yaw = math.atan2(
            move_vector[1],
            move_vector[0]
        )

        target_orientation = p.getQuaternionFromEuler(
            [0, 0, target_yaw]
        )

        if is_collision_free(robot_id, new_pos):

            p.resetBasePositionAndOrientation(
                robot_id,
                new_pos,
                target_orientation
            )

            state["last_pos"] = new_pos

            state["path_history"].append(new_pos)

        else:

            state["last_pos"] = current_pos

        state["last_orientation"] = target_orientation

        update_robot_package_position(
            robot_id,
            state["visible_package"]
        )

def check_and_display_dialogs():
    all_robots = [robot1, robot2, robot3, robot4]
    processed_pairs = set()
    current_time = time.time()
    
    for i, robot_id in enumerate(all_robots):
        for j, other_robot_id in enumerate(all_robots):
            if robot_id == other_robot_id or tuple(sorted([robot_id, other_robot_id])) in processed_pairs:
                continue
            pair = tuple(sorted([robot_id, other_robot_id]))
            processed_pairs.add(pair)
            
            state1 = robot_states[robot_id]
            state2 = robot_states[other_robot_id]
            if (state1["delivered"] and not state1["returning"]) or (state2["delivered"] and not state2["returning"]):
                continue
            
            pos1, _ = p.getBasePositionAndOrientation(robot_id)
            pos2, _ = p.getBasePositionAndOrientation(other_robot_id)
            if distance(pos1, pos2) < PASSING_RADIUS and not state1["is_paused"] and not state2["is_paused"]:
                if pair not in dialog_cooldowns or current_time - dialog_cooldowns[pair] >= DIALOG_COOLDOWN:
                    display_dialog_box(robot_id, other_robot_id)
                    dialog_cooldowns[pair] = current_time

def display_dialog_box(robot_id, other_robot_id):
    state1 = robot_states[robot_id]
    pos1, _ = p.getBasePositionAndOrientation(robot_id)
    current_time = time.strftime("%H:%M:%S", time.localtime())
    
    if robot_id not in message_boxes:
        message_boxes[robot_id] = []
    height_offset = len(message_boxes[robot_id]) * DIALOG_HEIGHT_OFFSET
    
    message = f"Hi, passing by [{current_time}]"
    box_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.35, 0.12, 0.01], rgbaColor=[0.9, 0.9, 0.9, 1.0])
    box_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=box_shape, basePosition=[pos1[0], pos1[1], pos1[2] + 0.5 + height_offset])
    text_id = p.addUserDebugText(text=message, textPosition=[pos1[0], pos1[1], pos1[2] + 0.52 + height_offset], textColorRGB=[0, 0, 0], textSize=0.6, lifeTime=MESSAGE_BOX_LIFETIME)
    message_boxes[robot_id].append(([box_id, text_id], time.time()))

def clean_message_boxes():
    current_time = time.time()
    for robot_id in list(message_boxes.keys()):
        to_remove = []
        for i, (ids, timestamp) in enumerate(message_boxes[robot_id]):
            if current_time - timestamp > MESSAGE_BOX_LIFETIME:
                for id in ids:
                    if id >= 0:
                        p.removeBody(id) if id in [b[0] for sublist in message_boxes.values() for b, _ in sublist] else p.removeUserDebugItem(id)
                to_remove.append(i)
        for i in reversed(to_remove):
            message_boxes[robot_id].pop(i)
        if not message_boxes[robot_id]:
            del message_boxes[robot_id]

def visualize_bases():
    for base_name, pos in bases.items():
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=BASE_RADIUS, length=0.05, rgbaColor=base_colors[base_name])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=[pos[0], pos[1], pos[2] - 0.02])

def create_legend():
    print("Robot Colors and Bases:")
    for robot_id in [robot1, robot2, robot3, robot4]:
        state = robot_states[robot_id]
        print(f"- {state['color']} Robot: Starts at {base_name_colors[state['current_base']]} Base")

def process_user_command(command):
    color_to_robot = {"green": robot1, "red": robot2, "blue": robot3, "yellow": robot4}
    
    try:
        assignments = command.lower().split(", ")
        if len(assignments) != 4:
            print("Please provide goals for all 4 robots (e.g., 'green to blue, red to yellow, blue to green, yellow to red')")
            return False
        
        targets = {}
        for assignment in assignments:
            parts = assignment.split(" to ")
            if len(parts) != 2:
                print(f"Invalid format in assignment: {assignment}")
                return False
            robot_color, target_color = parts
            robot_id = color_to_robot.get(robot_color)
            target_base = color_to_base.get(target_color)
            
            if not robot_id or not target_base:
                print(f"Invalid color in assignment: {assignment}. Use 'green', 'red', 'blue', or 'yellow'.")
                return False
            
            targets[robot_id] = target_base
        
        for robot_id in targets.keys():
            state = robot_states[robot_id]
            if not state["delivered"] or state["returning"]:
                print(f"{state['color']} Robot is already on a mission!")
                return False
            response = input(f"Should the {state['color']} robot stop at intermediate bases? (yes/no, default no): ").strip().lower()
            state["stop_at_intermediate"] = (response == "yes")
            if state["stop_at_intermediate"]:
                stop_bases_input = input(f"At which color bases should the {state['color']} robot stop? (e.g., red, yellow): ").strip().lower()
                stop_colors = [color.strip() for color in stop_bases_input.split(",") if color.strip()]
                state["intermediate_stop_bases"] = [color_to_base.get(color) for color in stop_colors if color_to_base.get(color)]
                print(f"{state['color']} Robot will stop at: {[base_name_colors[base] for base in state['intermediate_stop_bases']]}")
            else:
                state["intermediate_stop_bases"] = []
                print(f"{state['color']} Robot will not stop at intermediate bases.")
        
        for robot_id, target_base in targets.items():
            state = robot_states[robot_id]
            state["target_base"] = target_base
            state["delivered"] = False
            state["returning"] = False
            state["completion_time"] = None  
            state["stuck_counter"] = 0
            state["path_history"] = []
            state["has_traveled_message_printed"] = False
            state["intermediate_stops"] = 0
            state["stopped_bases"] = set()
            state["returned_message_printed"] = False
            state["current_intermediate_index"] = 0
            simulation_results["start_times"][state["name"]] = time.time()
            send_message(robot_id, f"Assigned to deliver from {base_name_colors[state['current_base']]} to {base_name_colors[target_base]}")
            print(f"{state['color']} Robot assigned to deliver from {base_name_colors[state['current_base']]} Base to {base_name_colors[target_base]} Base")
            pick_up_package(robot_id)
        
        return True
    except Exception as e:
        print(f"Error processing command: {e}")
        return False

def plot_simulation_results():
    robots = ["robot1", "robot2", "robot3", "robot4"]
    colors = ["Green", "Red", "Blue", "Yellow"]
    color_map = {"Green": "green", "Red": "red", "Blue": "blue", "Yellow": "yellow"}
    
    # Verify data is populated
    travel_times = [simulation_results["travel_times"].get(r, 0) for r in robots]
    stops = [simulation_results["intermediate_stops"].get(r, 0) for r in robots]
    avoidance_counts = [robot_states[robot_id]["avoidance_count"] for robot_id in [robot1, robot2, robot3, robot4]]
    
    print("Travel Times Data:", travel_times)
    print("Intermediate Stops Data:", stops)
    print("Avoidance Counts Data:", avoidance_counts)
    
    # 1. Robot Path Trajectories
    fig1 = plt.figure(figsize=(8, 8))
    ax1 = fig1.add_subplot(111)
    for robot_id in [robot1, robot2, robot3, robot4]:
        state = robot_states[robot_id]
        path = state["path_history"]
        if path:
            x, y = zip(*[(p[0], p[1]) for p in path])
            ax1.plot(x, y, color=color_map[state["color"]], linewidth=2, label=f"{state['color']} Robot")
            ax1.scatter(x[0], y[0], marker='o', s=100, color=color_map[state["color"]])
            ax1.scatter(x[-1], y[-1], marker='x', s=100, color=color_map[state["color"]])
    for base_name, pos in bases.items():
        ax1.scatter(pos[0], pos[1], marker='s', s=200, label=f"{base_name_colors[base_name]} Base", color=color_map[base_name_colors[base_name]])
    ax1.set_title("Robot Path Trajectories", fontsize=14)
    ax1.set_xlabel("X Position in m", fontsize=12)
    ax1.set_ylabel("Y Position in m", fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.text(0.05, 0.95, "Markers:\n• Circle: Start\n• X: End\n• Square: Base", transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    ax1.grid(True)
    plt.tight_layout()
    
    # 3. Obstacle Avoidance Frequency per Robot
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    ax2.bar(colors, avoidance_counts, color=[color_map[c] for c in colors])
    ax2.set_title("Obstacle Avoidance Frequency per Robot", fontsize=14)
    ax2.set_xlabel("Robots", fontsize=12)
    ax2.set_ylabel("Number of Avoidance Maneuvers", fontsize=12)
    ax2.grid(True)
    plt.tight_layout()
    
    # 4. Cumulative Distance Traveled Over Time
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    for robot_id in [robot1, robot2, robot3, robot4]:
        state = robot_states[robot_id]
        if state["distance_history"]:
            times, distances = zip(*state["distance_history"])
            times = [t - times[0] for t in times]
            ax3.plot(times, distances, label=f"{state['color']} Robot", color=color_map[state["color"]], linewidth=2)
    ax3.set_title("Cumulative Distance Traveled Over Time", fontsize=14)
    ax3.set_xlabel("Time (seconds)", fontsize=12)
    ax3.set_ylabel("Distance (m)", fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True)
    plt.tight_layout()
    
    # 5. Communication Hub Activity Over Time
    fig4 = plt.figure(figsize=(10, 6))
    ax4 = fig4.add_subplot(111)
    if communication_hub_activity:
        times, messages = zip(*communication_hub_activity)
        times = [t - times[0] for t in times]
        ax4.plot(times, messages, color='purple', linewidth=2)
    ax4.set_title("Communication Hub Activity Over Time", fontsize=14)
    ax4.set_xlabel("Time (seconds)", fontsize=12)
    ax4.set_ylabel("Number of Messages", fontsize=12)
    ax4.grid(True)
    plt.tight_layout()
    
    # Display all figures
    try:
        plt.show()
    except Exception as e:
        print(f"Error displaying plots: {e}")
        print("Saving plots to files instead.")
        fig1.savefig("path_trajectories.png")
        
        fig2.savefig("avoidance_frequency.png")
        fig3.savefig("distance_traveled.png")
        fig4.savefig("communication_activity.png")
        print("Plots saved as PNG files in the current directory.")

def run_simulation():
    global last_status_print  # Declare last_status_print as a global variable
    print("Initializing simulation with 4 robots and 4 delivery bases...")
    visualize_bases()
    create_legend()
    
    for base_name in bases:
        for _ in range(packages[base_name]):
            package_objects[base_name].append(create_package(base_name))
    
    all_robots = [robot1, robot2, robot3, robot4]
    
    print("\nEnter goals for all robots (e.g., 'green to blue, red to yellow, blue to green, yellow to red') or '0' to exit:")
    
    while True:
        command = input("> ")
        if command == '0':
            break
        
        if not command:
            continue
        
        if process_user_command(command):
            # Reset simulation state
            for robot_id in all_robots:
                state = robot_states[robot_id]
                state["delivered"] = False
                state["returning"] = False
                state["returned_message_printed"] = False
                state["avoidance_count"] = 0
                state["distance_history"] = []
                state["event_timeline"] = []
            
            # Reset communication hub activity
            global communication_hub_activity
            communication_hub_activity = []
            
            # Run simulation until all robots have delivered and returned
            while True:
                # Check if all robots are done
                all_done = True
                for robot in all_robots:
                    state = robot_states[robot]
                    if not (state["delivered"] and not state["returning"]):
                        all_done = False
                    print(f"Robot {state['color']}: delivered={state['delivered']}, returning={state['returning']}, target_base={state['target_base']}")
                
                all_returned = all(state["returned_message_printed"] for state in robot_states.values())
                
                if all_done or all_returned:
                    print("All robots have completed their tasks. Exiting simulation loop.")
                    break
                
                move_cars()
                for robot_id in all_robots:
                    robot_controller(robot_id)
                
                check_and_display_dialogs()
                clean_message_boxes()
                
                # Log communication hub activity
                current_time = time.time()
                communication_hub_activity.append((current_time, len(communication_hub["messages"])))
                
                p.stepSimulation()
                time.sleep(SIMULATION_DELAY)
                
                if current_time - last_status_print >= STATUS_PRINT_COOLDOWN:
                    print(f"Communication Hub: {len(communication_hub['messages'])} messages, "
                          f"{len(communication_hub['robot_status'])} robots active")
                    last_status_print = current_time
            
            print("\nSimulation Results:")
            for robot_id in all_robots:
                state = robot_states[robot_id]
                travel_time = simulation_results["travel_times"].get(state["name"], 0)
                stops = simulation_results["intermediate_stops"].get(state["name"], 0)
                print(f"{state['color']} Robot: Travel Time = {travel_time:.2f}s, Intermediate Stops = {stops}")
            plot_simulation_results()
            print("\nEnter new goals or '0' to exit:")

# Start the simulation
run_simulation()

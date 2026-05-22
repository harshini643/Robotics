# Multi-Agent Autonomous Robot Delivery Simulation

This repository contains a real-time autonomous robot delivery simulation developed using PyBullet and Python. The project simulates multiple robots operating in a shared environment while delivering packages, avoiding moving obstacles, and coordinating with each other dynamically.

Developed during my B.Tech in Computer Science (Specialization in AI) at Amrita Vishwa Vidyapeetham University.

---

## Project Overview

The simulation consists of four autonomous robots:

- Green Robot
- Red Robot
- Blue Robot
- Yellow Robot

Each robot:
- Picks up a package from its home base
- Navigates toward a target base
- Avoids moving cars and nearby robots
- Optionally stops at intermediate delivery stations
- Delivers the package
- Returns safely to its starting position

The project demonstrates:
- Autonomous robot navigation
- Multi-agent coordination
- Dynamic obstacle avoidance
- Deadlock recovery
- Real-time robotics simulation
- Visualization and analytics

---

## Technologies Used

- **Programming Language**: Python
- **Physics Simulation Engine**: PyBullet
- **Visualization**: Matplotlib
- **Numerical Computation**: NumPy
- **Utility Modules**: math, random, time

---

## Key Features

- Multi-robot autonomous delivery system
- Dynamic moving traffic simulation
- Real-time collision avoidance
- Robot-to-robot obstacle avoidance
- Deadlock detection and recovery
- Intermediate delivery stops
- Centralized communication hub
- Real-time movement analytics
- Smooth path planning and steering behavior

---

## Core Logic

### Autonomous Navigation

Each robot continuously calculates a direction toward its destination and updates its movement in real time.

### Obstacle Detection

Robots continuously detect:
- Moving cars
- Other robots

using distance-based calculations.

### Collision Avoidance

The movement system combines:
- Goal direction toward destination
- Repulsion direction away from nearby obstacles

This allows robots to safely navigate without collisions.

### Deadlock Recovery

If robots remain stuck for too long near obstacles or delivery bases, an emergency escape direction is generated automatically to recover movement.

### Dynamic Traffic Simulation

Cars move continuously across multiple lanes with looping road behavior to simulate real-world traffic obstacles.

---

## Analytics & Visualization

The project generates multiple graphs using Matplotlib:

- Robot Path Trajectories
- Obstacle Avoidance Frequency
- Cumulative Distance Traveled
- Communication Hub Activity

---

## Project Structure

```bash
robotics_simulation.py
README.md
```

---

## Getting Started

### Install Dependencies

```bash
pip install pybullet matplotlib numpy
```

### Run the Simulation

```bash
python robotics_simulation.py
```

---

## Example Input

```bash
green to blue, red to yellow, blue to green, yellow to red
```

---

## Requirements

- Python 3.8+
- PyBullet
- Matplotlib
- NumPy
- System with GUI support for PyBullet visualization

---


## License

This project is licensed under the MIT License.

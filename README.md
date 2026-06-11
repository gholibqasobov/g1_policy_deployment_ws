# g1_policy_deployment_ws

ROS 2 (Humble) deployment of the Isaac Lab **G1 flat-ground walking policy** inside Isaac Sim.

The `g1_fullbody_controller` node subscribes to the robot state, runs the trained policy at 50 Hz, and
publishes joint position targets back to Isaac Sim.

## What the policy expects

This controller is matched to the `Template-G1-Locomotion-v0` policy:

- **Observation (123)**, body frame, in this exact order (no normalization — fed raw):
  `base_lin_vel(3)` + `base_ang_vel(3)` + `projected_gravity(3)` + `velocity_command(3)` +
  `joint_pos_rel(37)` + `joint_vel(37)` + `last_action(37)`.
- **Action (37)** = raw policy output for **all** DoF (legs + arms + fingers).
  Joint target = `default_pos + 0.5 * action`.
- **Control rate = 50 Hz** (training used `sim.dt=0.005`, `decimation=4`).

The exact 37-joint order, per-joint defaults and action scaling are **not** hardcoded — they are loaded from
`policy_metadata.json`, which is generated from the live articulation (see below).

## ROS topics

| Direction | Topic | Type | Notes |
|---|---|---|---|
| in (synced) | `joint_states` | `sensor_msgs/JointState` | name + position + velocity; mapped into policy order by name |
| in (synced) | `imu` | `sensor_msgs/Imu` | `orientation` (body→world) + `angular_velocity` (body frame) |
| in (async) | `odom` | `nav_msgs/Odometry` | base linear velocity → `base_lin_vel` |
| in (async) | `cmd_vel` | `geometry_msgs/Twist` | `[linear.x, linear.y, angular.z]` command |
| out | `joint_command` | `sensor_msgs/JointState` | 37 named position targets |

`joint_states` and `imu` are time-synchronized (they must share a stamp). `odom`/`cmd_vel` are cached
asynchronously. Configure your Isaac Sim action graphs to publish the four inputs (recommended **50 Hz**)
and to consume `joint_command` with an Articulation Controller.

> **base_lin_vel frame:** by default the node treats `/odom` twist as **body frame** (REP-103). If your
> Isaac odometry graph publishes velocity in the **world** frame instead, launch with
> `odom_twist_in_body_frame:=False` and it will rotate it into the body frame using the IMU orientation.

## One-time setup

### 1. Generate the policy metadata (in the Isaac Lab / training env)

```bash
cd ~/g1_locomotion
python scripts/deploy/dump_policy_metadata.py \
    --task Template-G1-Locomotion-v0 \
    --out logs/rsl_rl/g1_locomotion_ppo/2026-06-10_21-48-08/exported/policy_metadata.json
```

This writes the authoritative joint order, defaults, `action_scale`, and obs layout next to the exported
`policy.pt`.

### 2. Point the controller at the policy + metadata

Either pass the log path directly:

```bash
ros2 launch g1_fullbody_controller g1_policy_controller.launch.py \
    policy_path:=$HOME/g1_locomotion/logs/rsl_rl/g1_locomotion_ppo/2026-06-10_21-48-08/exported/policy.pt
```

…or copy `policy.pt` **and** `policy_metadata.json` into
`src/g1_fullbody_controller/policy/` and use the default path. The metadata is loaded from the policy's
directory unless `metadata_path` is set.

## Build & run

```bash
cd ~/g1_policy_deployment_ws
pip install torch        # runtime dep, not a rosdep key
colcon build --packages-select g1_fullbody_controller
source install/setup.bash

ros2 launch g1_fullbody_controller g1_policy_controller.launch.py \
    policy_path:=/abs/path/to/exported/policy.pt
```

Drive it with a velocity command:

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}}'
```

You should see `/joint_command` stream 37 named targets at ~50 Hz and the robot walk forward.

## Launch parameters

| Param | Default | Description |
|---|---|---|
| `policy_path` | `policy/g1_policy.pt` | exported TorchScript policy |
| `metadata_path` | `<policy dir>/policy_metadata.json` | joint order / defaults / scaling |
| `decimation` | `1` | run policy every Nth tick (50 Hz sensors × 1 = 50 Hz control) |
| `odom_twist_in_body_frame` | `True` | set `False` if `/odom` twist is world-frame |
| `use_sim_time` | `True` | use the Isaac Sim clock |

## Troubleshooting

- **`policy_metadata.json not found`** — run step 1, or set `metadata_path`.
- **Robot drifts / poor velocity tracking** — flip `odom_twist_in_body_frame`; verify `/odom` actually
  carries a sensible base linear velocity.
- **Robot collapses immediately** — confirm the Articulation Controller applies `joint_command` as position
  targets (not effort) and that joint names match; the node maps by name, so a name mismatch silently leaves
  those joints at default.
- **No motion** — check the `joint_states`+`imu` stamps are equal (TimeSynchronizer needs matching stamps);
  if your graphs can't align them, tell me and we can switch to an approximate-time sync.

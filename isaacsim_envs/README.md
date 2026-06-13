# G1 policy — standalone Isaac Sim deployment scripts

Two self-contained Isaac Sim 5.0 scripts for deploying the trained G1 PPO walking policy
(`Template-G1-Locomotion-v0`, 37 DoF). Both launch with a plain `python script.py` once the Isaac Sim
conda env is active — no `python.sh`, no ROS sourcing.

```bash
conda activate env_isaaclab
```

The policy/metadata/env.yaml are read from the trained run:
`~/g1_locomotion/logs/rsl_rl/g1_locomotion_ppo/2026-06-10_21-48-08/` (override with `--policy-dir` /
`--policy-env`).

Both build the **123-d observation in the body frame** to match training:
`base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) | velocity_command(3) | joint_pos_rel(37) |
joint_vel(37) | last_action(37)`, with `target = default_joint_pos + 0.5*action`, 200 Hz physics /
50 Hz control. The trained effort caps (`effort_limit_sim`: legs/arms 300, ankles 20) are patched in
because the stock Isaac `config_loader` only reads `effort_limit` (which is null in the env.yaml).

> The body frame is the whole point: the earlier deployment exploded because OmniGraph nodes published
> IMU/odometry in the **world** frame while the policy was trained on body-frame ground truth.

## 1. `g1_policy_standalone.py` — direct deployment

The policy runs **inside Isaac** (H1-example style): Isaac reads ground truth, builds the observation,
runs `policy.pt`, applies joint targets. No ROS. The cleanest check that the policy itself walks.

```bash
python g1_policy_standalone.py                 # GUI
python g1_policy_standalone.py --headless      # no window
```

Drive from the focused viewport: `W/S` fwd/back, `A/D` strafe, `Q/E` (or `←/→`) turn, `SPACE` stop.
Tune the command magnitude with `--speed` (default 0.3). At startup it auto-walks forward at 0.3 m/s for
1 s to settle the gait, then accepts keyboard input.

## 2. `g1_controller_ros_standalone.py` — your controller + ROS, all in one process

A single command brings up Isaac Sim **plus** a body-frame ROS 2 bridge **plus** the real
`G1FullbodyController` (imported and spun in-process). Isaac publishes `/joint_states /imu /odom
/clock`; the controller runs the policy and publishes `/joint_command`; Isaac applies it. This
exercises the actual ROS deployment path.

```bash
python g1_controller_ros_standalone.py             # GUI
python g1_controller_ros_standalone.py --headless  # no window
```

Uses Isaac's **bundled** Python-3.11 `rclpy` (under `isaacsim.ros2.bridge/humble/rclpy`) — do **not**
source system ROS 2 Humble (it is built for Python 3.10 and will not import in the 3.11 env). A tiny
in-process `message_filters` shim is injected because the bundle omits it; the controller is imported
unmodified.

Drive it by publishing `geometry_msgs/Twist` to `/cmd_vel`. Any publisher in the **same conda env**
(so it shares Isaac's rclpy) works, e.g.:

```bash
conda activate env_isaaclab
python -c "import sys,os; \
import isaacsim, glob; \
sys.path.insert(0, glob.glob(os.path.join(os.path.dirname(isaacsim.__file__),'exts/isaacsim.ros2.bridge/humble/rclpy'))[0]); \
import rclpy; from geometry_msgs.msg import Twist; rclpy.init(); \
n=rclpy.create_node('teleop'); p=n.create_publisher(Twist,'cmd_vel',10); \
import time; t=Twist(); t.linear.x=0.5; \
[ (p.publish(t), time.sleep(0.1)) for _ in range(10000) ]"
```

(or run `teleop_twist_keyboard` from a system-ROS terminal — DDS bridges the two regardless of Python
version, as long as the `ROS_DOMAIN_ID` matches.)

## Anatomy of `g1_controller_ros_standalone.py` — recreate it yourself in GUI mode

This documents exactly what the script builds, so you can rebuild the same setup by hand (e.g. with
OmniGraph ROS2 Action Graph nodes) in the Isaac Sim GUI and run your controller against it.

### A. Robot configuration (`G1BridgeRobot`, `main()`)

| Item | Value / how |
|------|-------------|
| USD asset | `<assets_root>/Isaac/IsaacLab/Robots/Unitree/G1/g1.usd` (via `get_assets_root_path()`) |
| Stage prim path | `/World/G1` |
| Spawn pose | position `[0, 0, 0.74]` (trained base height), identity orientation |
| Articulation control | effort mode `force`, control mode **`position`** (PD position drive) |
| Joint stiffness (kp) | hip yaw/roll **150**, hip pitch/knee/torso **200**, ankles **20**, arms+hands **40** |
| Joint damping (kd) | legs/torso **5**, ankles **2**, arms+hands **10** |
| Effort caps (N·m) | legs/arms **300**, ankles **20** — set via `set_max_efforts` (env.yaml `effort_limit_sim`) |
| Default pose | the trained crouch from `policy_metadata.json` → `set_joints_default_state` + `set_joint_positions` |
| Physics rate | **200 Hz** (`physics_dt = 1/200 = 0.005 s`), render every 4 steps (`rendering_dt = 4/200`) |
| Scene | `add_default_ground_plane()` + a `DomeLight` at `/World/DomeLight` |

Gains/defaults come from the trained `env.yaml` via Isaac's `PolicyController.initialize()` /
`config_loader`; the script only patches the effort caps and the default pose on top.

**GUI equivalent:** drag `g1.usd` into the stage at `/World/G1`, place it at z=0.74, and on the
articulation set position drives with the stiffness/damping above (Property panel → Drive, per joint
group). Set physics dt to 0.005 (Physics Scene). Add a ground plane and a light.

### B. Sensor placement & frames (the part that must be exact)

There is **no separate sensor prim** in the script — all "sensor" values are read from the
**articulation root rigid body, i.e. G1's base/pelvis link** (what the floating-base pose refers to):
`get_world_pose()`, `get_linear_velocity()`, `get_angular_velocity()`. Everything is then expressed in
the **body frame** because that is what the policy was trained on. With `R_BI = quat_to_rot_matrix(q_world_body).T`
(world→body):

| Published field | Source | Frame / transform |
|-----------------|--------|-------------------|
| `/imu` `orientation` (w,x,y,z) | base world quaternion `q_world_body` | world pose, as-is |
| `/imu` `angular_velocity` | base angular velocity `ω_world` | **body**: `R_BI @ ω_world` |
| `/odom` `twist.twist.linear` | base linear velocity `v_world` | **body**: `R_BI @ v_world` |
| `/joint_states` `position`,`velocity` | all 37 DoF | joint space (order = `robot.dof_names`) |

`frame_id`s: `/imu` → `base_link`; `/odom` → frame `odom`, child `base_link`. The controller derives
`projected_gravity` itself from `/imu.orientation` (it does **not** come over a topic).

> **Critical:** the stock OmniGraph **Compute Odometry / Publish Odometry** node outputs the twist in the
> **world** frame, and that is exactly what blew the old `g1_flat_env.usd` up. Two correct options when
> recreating in GUI:
> - publish body-frame twist (rotate world→body as above), and keep the controller default
>   `odom_twist_in_body_frame:=True`; **or**
> - use the stock world-frame Compute Odometry node and launch the controller with
>   `odom_twist_in_body_frame:=False` (it then rotates world→body itself using `/imu.orientation`).
>
> The IMU sensor's angular velocity is already in its local (body) frame, so a stock **Publish IMU** node
> on the pelvis is fine as-is.

**GUI equivalent:** attach the IMU sensor and the Compute-Odometry "chassis" to the G1 **pelvis/base
link** prim (not the world). Make sure the odometry frame choice matches the controller param above.

### C. Nodes, topics & QoS

Two rclpy nodes are created and spun together in one `MultiThreadedExecutor` on a background thread;
the main thread steps physics.

**Bridge node** `g1_isaac_bridge` (`use_sim_time=True`):

| Direction | Topic | Type | QoS | Rate |
|-----------|-------|------|-----|------|
| publish | `/clock` | `rosgraph_msgs/Clock` | depth 10 | every physics step (200 Hz) |
| publish | `/joint_states` | `sensor_msgs/JointState` | RELIABLE · VOLATILE · KEEP_ALL | 200 Hz |
| publish | `/imu` | `sensor_msgs/Imu` | RELIABLE · VOLATILE · KEEP_ALL | 200 Hz |
| publish | `/odom` | `nav_msgs/Odometry` | RELIABLE · VOLATILE · KEEP_ALL | 200 Hz |
| subscribe | `/joint_command` | `sensor_msgs/JointState` | RELIABLE · VOLATILE · KEEP_ALL | applied next step |

`/joint_states` and `/imu` are stamped with the **same** sim time each step — the controller pairs them
with a `TimeSynchronizer`, which needs exact-matching stamps. Incoming `/joint_command` is name-mapped
into DOF order, clamped to joint limits, and applied via `ArticulationController`.

**Controller node** `g1_policy_controller` (your `G1FullbodyController`, imported in-process) is
constructed with `rclpy.init(args=["--ros-args", "-p", "policy_path:=…", "-p", "warmup_sec:=0.0",
"-p", "odom_twist_in_body_frame:=true", "-p", "use_sim_time:=true"])`. It subscribes `/joint_states`
(+`/imu` time-synced), `/odom`, `/cmd_vel`, and publishes `/joint_command` at 50 Hz.

**GUI equivalent (OmniGraph ROS2 Action Graph on `/World/G1`):** *On Playback Tick* →
`ROS2 Publish Clock` (`/clock`); `ROS2 Publish Joint State` (articulation `/World/G1` → `/joint_states`);
`ROS2 Subscribe Joint State` (`/joint_command`) → `Articulation Controller` (`/World/G1`);
`ROS2 Publish Imu` (pelvis IMU → `/imu`); `Isaac Compute Odometry` (chassis = pelvis) →
`ROS2 Publish Odometry` (`/odom`). Then run your controller from a **sourced ROS 2 Humble terminal**
(`ros2 run g1_fullbody_controller g1_policy_controller --ros-args -p policy_path:=… -p use_sim_time:=true
-p odom_twist_in_body_frame:=<match your odom frame>`). DDS bridges the GUI (Isaac's rclpy) and your
terminal controller (system py3.10 rclpy) — only the `ROS_DOMAIN_ID` and RMW vendor must match.

### D. Per-step loop order (`main()`)

Each physics step the main thread: (1) `bridge.publish_state(sim_ns)` → publishes `/clock`,
`/joint_states`, `/imu`, `/odom` with one shared stamp; (2) reads the latest cached `/joint_command`;
(3) applies it to the articulation — or, during the startup warmup, holds the default pose and
`_freeze_base(...)` until the first `/joint_command` arrives; (4) `world.step()`.

## Startup stabilization (why the robot doesn't fall on spawn)

The G1's ankle actuators are soft (`kp=20`), so a *passive* PD hold of the trained crouch is not
statically balanced — only the **active policy** keeps it upright. Both scripts therefore:

1. Run control from the **first physics step** (Script 1 calls `policy.forward()` every step, H1-example
   style; Script 2 hands off to the controller as soon as it commands), and
2. **Pin the floating base** (re-write its pose + zero its velocity each step) during a short warmup, then
   release — Script 1 for ~0.3 s while gains settle, Script 2 until the first `/joint_command` arrives
   (2 s safety timeout). This covers the ROS startup latency (DDS discovery + round-trip).

Keep the spawn height at the trained **z≈0.74**. Do **not** lift the robot — a higher spawn just
free-falls and crashes.

## Notes

- First launch downloads the G1 USD from the Isaac asset server; subsequent runs use the local cache.
- The `config_loader` "default position not found, setting to 0" warnings are harmless — those joints
  genuinely default to 0, and the scripts use the metadata defaults regardless.
- `isaacsim_envs/g1_flat_env.usd` is the **old, broken** world-frame OmniGraph setup. Neither script
  opens it; they build a fresh ground-plane + dome-light scene.

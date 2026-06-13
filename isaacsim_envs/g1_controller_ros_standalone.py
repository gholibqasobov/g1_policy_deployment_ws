#!/usr/bin/env python3
"""All-in-one standalone Isaac Sim + ROS 2 deployment driven by the user's G1FullbodyController.

A single `python g1_controller_ros_standalone.py` brings up everything:

    * Isaac Sim spawns the G1 with the trained gains / effort caps / default pose,
    * a body-frame ROS 2 bridge publishes ground truth (/joint_states /imu /odom /clock) and applies
      the controller's /joint_command back to the articulation,
    * the user's `G1FullbodyController` (which owns the policy) is imported and spun in-process.

This validates the real ROS deployment path. The previous deployment exploded because hand-built
OmniGraph nodes published IMU/odometry in the WORLD frame while the policy was trained on BODY-frame
ground truth -- this bridge publishes everything in the body frame (see memory g1-deploy-obs-frames.md).

Run (with the Isaac Sim conda env active -- do NOT source system ROS 2; this uses Isaac's bundled,
Python-3.11 rclpy):

    conda activate env_isaaclab
    python g1_controller_ros_standalone.py                # GUI window
    python g1_controller_ros_standalone.py --headless     # no window

Then drive it from another terminal (also in the conda env so it shares Isaac's rclpy):
    python g1_controller_ros_standalone.py --teleop        # prints the teleop hint, or:
    # publish geometry_msgs/Twist to /cmd_vel however you like, e.g. teleop_twist_keyboard.
"""

import argparse
import json
import os
import sys
import threading
import types

# --- defaults -------------------------------------------------------------------------------------
_RUN_DIR = os.path.expanduser(
    "~/g1_locomotion/logs/rsl_rl/g1_locomotion_ppo/2026-06-10_21-48-08"
)
_CONTROLLER_DIR = os.path.expanduser(
    "~/g1_policy_deployment_ws/src/g1_fullbody_controller/g1_fullbody_controller"
)

parser = argparse.ArgumentParser(description="Standalone Isaac Sim + ROS2 + G1FullbodyController.")
parser.add_argument("--headless", action="store_true", help="Run without a viewport window.")
parser.add_argument("--policy-dir", default=os.path.join(_RUN_DIR, "exported"),
                    help="Directory holding policy.pt + policy_metadata.json.")
parser.add_argument("--policy-env", default=os.path.join(_RUN_DIR, "params", "env.yaml"),
                    help="Trained env.yaml (gains / limits / timing).")
parser.add_argument("--robot-usd", default=None, help="Override the G1 USD path.")
parser.add_argument("--controller-dir", default=_CONTROLLER_DIR,
                    help="Directory containing g1_policy_controller.py.")
args = parser.parse_args()


def _scrub_system_ros():
    """Remove globally-sourced ROS 2 Humble (Python 3.10) paths from the environment.

    This conda env is Python 3.11; if the shell has sourced /opt/ros/humble (and assorted py3.10
    workspaces), their paths land on PYTHONPATH/AMENT_PREFIX_PATH/sys.path and Isaac's ros2 bridge
    tries to import the system 3.10 rclpy into the 3.11 interpreter -> ImportError. Dropping these
    makes the bridge fall back to its bundled, Python-3.11 rclpy.
    """
    def _clean(value):
        if not value:
            return value
        return os.pathsep.join(
            p for p in value.split(os.pathsep)
            if p and "/opt/ros/" not in p and "python3.10" not in p
        )

    for var in ("PYTHONPATH", "LD_LIBRARY_PATH", "CMAKE_PREFIX_PATH"):
        if var in os.environ:
            os.environ[var] = _clean(os.environ[var])
    # a populated AMENT_PREFIX_PATH is how the bridge detects "ROS is sourced" -> drop it entirely
    os.environ.pop("AMENT_PREFIX_PATH", None)
    sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p and "python3.10" not in p]


_scrub_system_ros()

# --- boot Isaac Sim BEFORE any other isaac import -------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.robot.policy.examples")
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()


# --- make Isaac's bundled (Python 3.11) ROS 2 packages importable --------------------------------
def _bundled_ros_python_path():
    """Locate the rclpy/message python packages shipped inside the ros2 bridge extension."""
    distro = os.environ.get("ROS_DISTRO", "humble")
    candidates = []
    try:
        import omni.kit.app
        mgr = omni.kit.app.get_app().get_extension_manager()
        ext_path = mgr.get_extension_path_by_module("isaacsim.ros2.bridge")
        if ext_path:
            candidates.append(os.path.join(ext_path, distro, "rclpy"))
    except Exception:
        pass
    import isaacsim
    candidates.append(os.path.join(
        os.path.dirname(isaacsim.__file__), "exts", "isaacsim.ros2.bridge", distro, "rclpy"))
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise RuntimeError(
        f"Could not find Isaac's bundled rclpy for ROS_DISTRO={distro}; tried: {candidates}")


_ros_py = _bundled_ros_python_path()
if _ros_py not in sys.path:
    sys.path.insert(0, _ros_py)


def _install_message_filters_shim():
    """The controller needs message_filters, which Isaac's bundle omits. Provide a minimal one.

    Implements just Subscriber + TimeSynchronizer with exact header-stamp matching, which is all the
    controller uses; the bridge publishes /joint_states and /imu with identical stamps each step.
    """
    if "message_filters" in sys.modules:
        return
    mf = types.ModuleType("message_filters")

    class Subscriber:
        def __init__(self, node, msg_type, topic, qos_profile=10, **kwargs):
            self._cbs = []
            node.create_subscription(msg_type, topic, self._handle, qos_profile)

        def registerCallback(self, cb):
            self._cbs.append(cb)

        def _handle(self, msg):
            for cb in self._cbs:
                cb(msg)

    class TimeSynchronizer:
        def __init__(self, subscribers, queue_size):
            self._subs = subscribers
            self._qs = queue_size
            self._cbs = []
            self._buf = [dict() for _ in subscribers]
            for i, sub in enumerate(subscribers):
                sub.registerCallback(lambda msg, i=i: self._add(i, msg))

        def registerCallback(self, cb):
            self._cbs.append(cb)

        @staticmethod
        def _key(msg):
            s = msg.header.stamp
            return (s.sec, s.nanosec)

        def _add(self, idx, msg):
            key = self._key(msg)
            self._buf[idx][key] = msg
            if all(key in b for b in self._buf):
                msgs = [b.pop(key) for b in self._buf]
                for cb in self._cbs:
                    cb(*msgs)
            buf = self._buf[idx]
            if len(buf) > self._qs:  # trim stale unmatched messages
                for old in sorted(buf)[:-self._qs]:
                    buf.pop(old, None)

    mf.Subscriber = Subscriber
    mf.TimeSynchronizer = TimeSynchronizer
    sys.modules["message_filters"] = mf


_install_message_filters_shim()

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from builtin_interfaces.msg import Time as TimeMsg  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from sensor_msgs.msg import Imu, JointState  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.prims import create_prim  # noqa: E402
from isaacsim.core.utils.rotations import quat_to_rot_matrix  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.robot.policy.examples.controllers import PolicyController  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

# match the controller's QoS so subscriptions connect (RELIABLE / VOLATILE / KEEP_ALL)
SIM_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_ALL,
)


class G1BridgeRobot(PolicyController):
    """Spawns the G1 with the trained gains/limits/defaults. Does NOT run the policy.

    The user's controller owns the policy; this class only applies physics config and exposes
    helpers for the ROS bridge.
    """

    def __init__(self, prim_path, policy_dir, policy_env, usd_path=None, position=None):
        assets_root = get_assets_root_path()
        if usd_path is None:
            usd_path = assets_root + "/Isaac/IsaacLab/Robots/Unitree/G1/g1.usd"
        super().__init__("g1", prim_path, None, usd_path, position, None)
        # load_policy is reused only to parse env.yaml (gains/limits/timing); the jit is never run.
        self.load_policy(os.path.join(policy_dir, "policy.pt"), policy_env)

        with open(os.path.join(policy_dir, "policy_metadata.json")) as f:
            meta = json.load(f)
        self.policy_joint_names = list(meta["joint_names"])
        self.default_pos_policy = np.asarray(meta["default_joint_pos"], dtype=np.float32)
        self._num_joints = int(meta["num_joints"])
        self._lower = None
        self._upper = None

    def initialize(self):
        super().initialize(set_articulation_props=False)
        self.dof_names = list(self.robot.dof_names)
        self._name_to_dof = {n: i for i, n in enumerate(self.dof_names)}
        missing = [n for n in self.policy_joint_names if n not in self._name_to_dof]
        if missing:
            raise RuntimeError(f"Policy joints absent from the G1 USD articulation: {missing}")
        self._dof_index_for_policy = np.array(
            [self._name_to_dof[n] for n in self.policy_joint_names], dtype=np.int64
        )

        # env.yaml uses effort_limit_sim (legs/arms 300, feet 20) with effort_limit=null, which the
        # stock config_loader ignores -> apply the trained PhysX effort caps explicitly.
        efforts = np.array([20.0 if "ankle" in n else 300.0 for n in self.dof_names], dtype=np.float32)
        self.robot._articulation_view.set_max_efforts(efforts)

        try:  # joint limits for clamping incoming /joint_command
            limits = np.asarray(self.robot._articulation_view.get_dof_limits())
            limits = limits.reshape(-1, self._num_joints, 2)
            self._lower = limits[0, :, 0]
            self._upper = limits[0, :, 1]
        except Exception:
            self._lower = self._upper = None

    def default_pose_dof(self):
        tgt = np.zeros(self._num_joints, dtype=np.float32)
        tgt[self._dof_index_for_policy] = self.default_pos_policy
        return tgt


class G1RosBridge(Node):
    """Publishes G1 ground truth in the body frame and applies /joint_command to the articulation."""

    def __init__(self, robot: G1BridgeRobot):
        super().__init__("g1_isaac_bridge")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
        self.robot = robot
        self._lock = threading.Lock()
        self._latest_cmd = None  # np array in DOF order

        self._pub_js = self.create_publisher(JointState, "joint_states", SIM_QOS)
        self._pub_imu = self.create_publisher(Imu, "imu", SIM_QOS)
        self._pub_odom = self.create_publisher(Odometry, "odom", SIM_QOS)
        self._pub_clock = self.create_publisher(Clock, "clock", 10)
        self.create_subscription(JointState, "joint_command", self._on_command, SIM_QOS)

    def _on_command(self, msg: JointState):
        """Map a policy-order /joint_command into DOF order, clamp, and cache it."""
        cmd = self.robot.default_pose_dof()
        for name, pos in zip(msg.name, msg.position):
            idx = self.robot._name_to_dof.get(name)
            if idx is not None:
                cmd[idx] = pos
        if self.robot._lower is not None:
            cmd = np.clip(cmd, self.robot._lower, self.robot._upper)
        with self._lock:
            self._latest_cmd = cmd

    def get_command(self):
        with self._lock:
            return None if self._latest_cmd is None else self._latest_cmd.copy()

    def publish_state(self, sim_time_ns: int):
        stamp = TimeMsg(sec=int(sim_time_ns // 1_000_000_000),
                        nanosec=int(sim_time_ns % 1_000_000_000))

        self._pub_clock.publish(Clock(clock=stamp))

        # body-frame transform from the articulation's world pose
        lin_vel_I = self.robot.robot.get_linear_velocity()
        ang_vel_I = self.robot.robot.get_angular_velocity()
        _, q_IB = self.robot.robot.get_world_pose()  # (w, x, y, z)
        R_BI = quat_to_rot_matrix(q_IB).transpose()  # world -> body
        ang_vel_b = R_BI @ ang_vel_I
        lin_vel_b = R_BI @ lin_vel_I

        jp = self.robot.robot.get_joint_positions()
        jv = self.robot.robot.get_joint_velocities()
        js = JointState()
        js.header.stamp = stamp
        js.name = self.robot.dof_names
        js.position = jp.astype(float).tolist()
        js.velocity = jv.astype(float).tolist()
        self._pub_js.publish(js)

        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = "base_link"
        imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z = (
            float(q_IB[0]), float(q_IB[1]), float(q_IB[2]), float(q_IB[3]))
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = map(float, ang_vel_b)
        self._pub_imu.publish(imu)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = (
            map(float, lin_vel_b))  # body frame -> controller default odom_twist_in_body_frame=True
        self._pub_odom.publish(odom)


def _freeze_base(art, pin_pos, pin_quat):
    """Hold the floating base fixed for a warmup step (articulation-safe pin).

    PhysX does not honor a kinematic flag on an articulation root, so we re-write the base pose and
    zero its velocity each step instead -- the robot can't fall while the control loop spins up.
    """
    art.set_world_pose(pin_pos, pin_quat)
    art.set_linear_velocity(np.zeros(3))
    art.set_angular_velocity(np.zeros(3))


def main():
    world = World(physics_dt=1.0 / 200.0, rendering_dt=4.0 / 200.0, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    create_prim("/World/DomeLight", "DomeLight", attributes={"inputs:intensity": 1000.0})

    robot = G1BridgeRobot(
        prim_path="/World/G1",
        policy_dir=args.policy_dir,
        policy_env=args.policy_env,
        usd_path=args.robot_usd,
        position=np.array([0.0, 0.0, 0.74]),  # trained spawn height -- do NOT lift (free-fall = worse)
    )
    world.reset()
    robot.initialize()
    robot.post_reset()
    # place in the trained crouch with zero velocity so the controller's first observation is in-distribution
    default_dof = robot.default_pose_dof()
    robot.robot.set_joints_default_state(positions=default_dof)
    robot.robot.set_joint_positions(default_dof)
    robot.robot.set_joint_velocities(np.zeros(robot._num_joints, dtype=np.float32))
    pin_pos, pin_quat = robot.robot.get_world_pose()

    # ---- ROS: bridge node + the user's controller, spun together in a background thread ----------
    policy_pt = os.path.join(args.policy_dir, "policy.pt")
    rclpy.init(args=[
        "--ros-args",
        "-p", f"policy_path:={policy_pt}",
        "-p", "warmup_sec:=0.0",            # Isaac holds the default pose during settle instead
        "-p", "odom_twist_in_body_frame:=true",
        "-p", "use_sim_time:=true",
    ])
    bridge = G1RosBridge(robot)

    if args.controller_dir not in sys.path:
        sys.path.insert(0, args.controller_dir)
    from g1_policy_controller import G1FullbodyController
    controller = G1FullbodyController()

    executor = MultiThreadedExecutor()
    executor.add_node(bridge)
    executor.add_node(controller)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    dt = 1.0 / 200.0
    dt_ns = int(round(dt * 1e9))
    # The crouch is NOT statically stable (ankle kp=20); only the active policy balances it. Pin the base
    # until the controller is actively commanding (first /joint_command), so it can't fall while the ROS
    # loop spins up (DDS discovery + bridge->controller->bridge round-trip). 2 s safety timeout.
    max_pin_steps = int(2.0 / dt)
    released = False

    print("[g1_controller_ros_standalone] running. Publish geometry_msgs/Twist to /cmd_vel to drive "
          "(e.g. teleop_twist_keyboard from a conda-env terminal).")
    step = 0
    try:
        while simulation_app.is_running():
            bridge.publish_state(step * dt_ns)   # publish ground truth every step so the controller can compute
            cmd = bridge.get_command()
            if released and cmd is not None:
                robot.robot.apply_action(ArticulationAction(joint_positions=cmd))
            else:
                robot.robot.apply_action(ArticulationAction(joint_positions=default_dof))
                _freeze_base(robot.robot, pin_pos, pin_quat)  # hold upright until handover
            if not released and (cmd is not None or step >= max_pin_steps):
                released = True
            world.step(render=not args.headless)
            step += 1
    finally:
        executor.shutdown()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()

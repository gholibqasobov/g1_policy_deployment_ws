#!/usr/bin/env python3
"""Standalone Isaac Sim deployment of the trained G1 PPO walking policy (direct, H1-example style).

Isaac Sim reads the robot's ground-truth state, builds the 123-d body-frame observation exactly as in
the trained ``Template-G1-Locomotion-v0`` env, runs the exported TorchScript ``policy.pt`` in-process,
and applies joint position targets. No ROS, no external controller -- this is the end-to-end sanity
check that the policy itself walks.

Run (with the Isaac Sim conda env active):

    conda activate env_isaaclab
    python g1_policy_standalone.py                 # GUI window
    python g1_policy_standalone.py --headless      # no window

Drive it from the Isaac viewport (window focused):
    W / S : forward / backward      A / D : strafe left / right
    Q / E : turn left / right       SPACE : stop      (arrow keys also work)

Observation layout (123) -- body frame, must match training exactly:
    base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) | velocity_command(3)
    | joint_pos_rel(37) | joint_vel(37) | last_action(37)
Action(37): target = default_joint_pos + 0.5 * action.
"""

import argparse
import json
import os

# --- defaults -------------------------------------------------------------------------------------
_RUN_DIR = os.path.expanduser(
    "~/g1_locomotion/logs/rsl_rl/g1_locomotion_ppo/2026-06-10_21-48-08"
)

parser = argparse.ArgumentParser(description="Direct standalone G1 policy deployment in Isaac Sim.")
parser.add_argument("--headless", action="store_true", help="Run without a viewport window.")
parser.add_argument("--policy-dir", default=os.path.join(_RUN_DIR, "exported"),
                    help="Directory holding policy.pt + policy_metadata.json.")
parser.add_argument("--policy-env", default=os.path.join(_RUN_DIR, "params", "env.yaml"),
                    help="Trained env.yaml (gains / limits / timing).")
parser.add_argument("--robot-usd", default=None,
                    help="Override the G1 USD path (defaults to the Isaac asset server).")
parser.add_argument("--speed", type=float, default=0.3, help="Commanded speed magnitude (m/s, rad/s).")
args = parser.parse_args()

# --- boot Isaac Sim BEFORE any other isaac import -------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.appwindow  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.prims import create_prim  # noqa: E402
from isaacsim.core.utils.rotations import quat_to_rot_matrix  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

enable_extension("isaacsim.robot.policy.examples")
from isaacsim.robot.policy.examples.controllers import PolicyController  # noqa: E402


class G1FlatTerrainPolicy(PolicyController):
    """Runs the trained G1 flat-terrain locomotion policy directly from Isaac ground truth.

    Mirrors Isaac's H1FlatTerrainPolicy but for G1's 37 DoF / 123-d observation, and works in the
    policy's joint order (from policy_metadata.json) regardless of the USD's DOF ordering.
    """

    def __init__(self, prim_path, policy_dir, policy_env, usd_path=None, name="g1",
                 position=None, orientation=None):
        assets_root = get_assets_root_path()
        if usd_path is None:
            usd_path = assets_root + "/Isaac/IsaacLab/Robots/Unitree/G1/g1.usd"
        super().__init__(name, prim_path, None, usd_path, position, orientation)

        # load the exported TorchScript policy + the trained env.yaml (gains/limits/timing)
        self.load_policy(os.path.join(policy_dir, "policy.pt"), policy_env)

        # joint order / defaults / scaling come from the metadata dumped next to the policy
        with open(os.path.join(policy_dir, "policy_metadata.json")) as f:
            meta = json.load(f)
        self.policy_joint_names = list(meta["joint_names"])
        self.default_pos_policy = np.asarray(meta["default_joint_pos"], dtype=np.float32)
        self._action_scale = float(meta["action_scale"])
        self._obs_dim = int(meta["obs_dim"])
        self._num_joints = int(meta["num_joints"])

        self._previous_action = np.zeros(self._num_joints, dtype=np.float32)
        self.action = np.zeros(self._num_joints, dtype=np.float32)
        self._policy_counter = 0
        self._dof_index_for_policy = None  # filled in initialize()

    def initialize(self):
        """Set gains/limits/defaults from env.yaml, then patch in the trained effort caps."""
        # set_articulation_props=False -> keep the USD's articulation root settings (matches H1 example)
        super().initialize(set_articulation_props=False)

        # build the (policy-order -> DOF-order) index map; the USD DOF order need not match the policy
        dof_names = list(self.robot.dof_names)
        missing = [n for n in self.policy_joint_names if n not in dof_names]
        if missing:
            raise RuntimeError(f"Policy joints absent from the G1 USD articulation: {missing}")
        self._dof_index_for_policy = np.array(
            [dof_names.index(n) for n in self.policy_joint_names], dtype=np.int64
        )
        if not np.array_equal(self._dof_index_for_policy, np.arange(self._num_joints)):
            carb.log_warn("G1 USD DOF order differs from policy order; remapping by joint name.")

        # env.yaml uses effort_limit_sim (legs/arms 300, feet 20) with effort_limit=null, which the
        # stock config_loader ignores -> apply the trained PhysX effort caps explicitly.
        efforts = np.array([20.0 if "ankle" in n else 300.0 for n in dof_names], dtype=np.float32)
        self.robot._articulation_view.set_max_efforts(efforts)

    def default_pose_dof(self):
        """Default joint pose scattered into the articulation's DOF order."""
        tgt_dof = np.zeros(self._num_joints, dtype=np.float32)
        tgt_dof[self._dof_index_for_policy] = self.default_pos_policy
        return tgt_dof

    def _compute_observation(self, command):
        """Assemble the 123-d body-frame observation (training order)."""
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        _, q_IB = self.robot.get_world_pose()

        R_BI = quat_to_rot_matrix(q_IB).transpose()  # world -> body
        lin_vel_b = R_BI @ lin_vel_I
        ang_vel_b = R_BI @ ang_vel_I
        gravity_b = R_BI @ np.array([0.0, 0.0, -1.0])

        jp_dof = self.robot.get_joint_positions()
        jv_dof = self.robot.get_joint_velocities()
        jp_policy = jp_dof[self._dof_index_for_policy]
        jv_policy = jv_dof[self._dof_index_for_policy]

        obs = np.zeros(self._obs_dim, dtype=np.float32)
        n = self._num_joints
        obs[0:3] = lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = command
        obs[12:12 + n] = jp_policy - self.default_pos_policy
        obs[12 + n:12 + 2 * n] = jv_policy
        obs[12 + 2 * n:12 + 3 * n] = self._previous_action
        return obs

    def forward(self, dt, command):
        """Recompute the action every ``decimation`` ticks; always apply the held target."""
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        tgt_policy = self.default_pos_policy + self.action * self._action_scale
        tgt_dof = np.zeros(self._num_joints, dtype=np.float32)
        tgt_dof[self._dof_index_for_policy] = tgt_policy
        self.robot.apply_action(ArticulationAction(joint_positions=tgt_dof))
        self._policy_counter += 1


# --- keyboard teleop ------------------------------------------------------------------------------
class KeyboardCommander:
    """Maps held keys in the Isaac viewport to a (vx, vy, wz) command."""

    def __init__(self, speed):
        self._speed = speed
        self._pressed = set()
        self.command = np.zeros(3, dtype=np.float32)
        self._bindings = {
            carb.input.KeyboardInput.W: (0, +1.0), carb.input.KeyboardInput.UP: (0, +1.0),
            carb.input.KeyboardInput.S: (0, -1.0), carb.input.KeyboardInput.DOWN: (0, -1.0),
            carb.input.KeyboardInput.A: (1, +1.0), carb.input.KeyboardInput.D: (1, -1.0),
            carb.input.KeyboardInput.Q: (2, +1.0), carb.input.KeyboardInput.LEFT: (2, +1.0),
            carb.input.KeyboardInput.E: (2, -1.0), carb.input.KeyboardInput.RIGHT: (2, -1.0),
        }
        appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._sub = self._input.subscribe_to_keyboard_events(
            appwindow.get_keyboard(), self._on_event
        )

    def _on_event(self, event, *args):
        et = event.type
        if et == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.SPACE:
                self._pressed.clear()
            else:
                self._pressed.add(event.input)
        elif et == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(event.input)
        self._recompute()
        return True

    def _recompute(self):
        cmd = np.zeros(3, dtype=np.float32)
        for key in self._pressed:
            axis_sign = self._bindings.get(key)
            if axis_sign is not None:
                axis, sign = axis_sign
                cmd[axis] += sign * self._speed
        self.command = cmd


def _freeze_base(art, pin_pos, pin_quat):
    """Hold the floating base fixed for a warmup step (articulation-safe pin).

    PhysX does not honor a kinematic flag on an articulation root, so we re-write the base pose and
    zero its velocity each step instead -- the robot can't fall while the policy/gains spin up.
    """
    art.set_world_pose(pin_pos, pin_quat)
    art.set_linear_velocity(np.zeros(3))
    art.set_angular_velocity(np.zeros(3))


def main():
    world = World(physics_dt=1.0 / 200.0, rendering_dt=4.0 / 200.0, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    create_prim("/World/DomeLight", "DomeLight", attributes={"inputs:intensity": 1000.0})

    robot = G1FlatTerrainPolicy(
        prim_path="/World/G1",
        policy_dir=args.policy_dir,
        policy_env=args.policy_env,
        usd_path=args.robot_usd,
        position=np.array([0.0, 0.0, 0.74]),  # trained spawn height -- do NOT lift (free-fall = worse)
    )

    world.reset()
    robot.initialize()
    robot.post_reset()
    # place in the trained crouch with zero velocity so the first observation is in-distribution
    default_dof = robot.default_pose_dof()
    robot.robot.set_joints_default_state(positions=default_dof)
    robot.robot.set_joint_positions(default_dof)
    robot.robot.set_joint_velocities(np.zeros(robot._num_joints, dtype=np.float32))
    pin_pos, pin_quat = robot.robot.get_world_pose()

    commander = KeyboardCommander(args.speed) if not args.headless else None
    dt = 1.0 / 200.0
    # The crouch is NOT statically stable (ankle kp=20); only the active policy balances it. So run the
    # policy from frame 0 (H1-example pattern) and just pin the base briefly while gains/joints settle.
    warmup_steps = int(0.3 / dt)
    # Drive a small forward command (0.3 m/s) for the first 1 s so the robot walks itself into a stable
    # gait before handing control to the keyboard.
    startup_steps = int(0.2 / dt)
    startup_command = np.array([0.3, 0.0, 0.0], dtype=np.float32)

    print("[g1_policy_standalone] running. Focus the viewport: W/S A/D Q/E (arrows) to drive, SPACE to stop.")
    step = 0
    while simulation_app.is_running():
        if step < startup_steps:
            command = startup_command                    # forced forward command for 1 s to settle the gait
        else:
            command = commander.command if commander is not None else np.zeros(3, dtype=np.float32)
        robot.forward(dt, command)                       # policy active every step
        if step < warmup_steps:
            _freeze_base(robot.robot, pin_pos, pin_quat)  # hold base upright during settle, then release
        world.step(render=not args.headless)
        step += 1

    simulation_app.close()


if __name__ == "__main__":
    main()

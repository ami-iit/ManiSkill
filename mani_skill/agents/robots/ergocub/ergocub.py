import os
import tempfile

import idyntree.bindings as idyn
import numpy as np
import resolve_robotics_uri_py as rru
import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDBaseForwardVelControllerConfig,
    PDEEPosControllerConfig,
    PDJointPosControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.controllers.pd_base_vel import PDBaseVelControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent(override=True)
class ErgoCub(BaseAgent):
    uid = "ergocub"
    urdf_path_original = rru.resolve_robotics_uri(
        "package://ergoCub/robots/ergoCubSN001/model.urdf"
    )

    # Joint names categorized by body parts
    joint_names = [
        "torso_roll",
        "torso_pitch",
        "torso_yaw",
        "neck_pitch",
        "l_shoulder_pitch",
        "r_shoulder_pitch",
        "neck_roll",
        "l_shoulder_roll",
        "r_shoulder_roll",
        "neck_yaw",
        "camera_tilt",
        "l_shoulder_yaw",
        "r_shoulder_yaw",
        "l_elbow",
        "r_elbow",
        "l_wrist_yaw",
        "r_wrist_yaw",
        "l_wrist_roll",
        "r_wrist_roll",
        "l_wrist_pitch",
        "r_wrist_pitch",
        "l_index_add",
        "l_middle_prox",
        "l_pinkie_prox",
        "l_ring_prox",
        "l_thumb_add",
        "r_index_add",
        "r_middle_prox",
        "r_pinkie_prox",
        "r_ring_prox",
        "r_thumb_add",
        "l_index_prox",
        "l_middle_dist",
        "l_pinkie_dist",
        "l_ring_dist",
        "l_thumb_prox",
        "r_index_prox",
        "r_middle_dist",
        "r_pinkie_dist",
        "r_ring_dist",
        "r_thumb_prox",
        "l_index_dist",
        "l_thumb_dist",
        "r_index_dist",
        "r_thumb_dist",
    ]

    model_loader = idyn.ModelLoader()
    model_loader.loadReducedModelFromFile(str(urdf_path_original), joint_names)
    model = model_loader.model()
    root_link = model.getDefaultBaseLink()

    noInertiaLink = idyn.Link()
    noInertiaLink.setInertia(idyn.SpatialInertia.Zero())

    world = model.addLink("world", noInertiaLink)
    root_arm_1_link_1 = model.addLink("root_arm_1_link_1", noInertiaLink)
    root_arm_1_link_2 = model.addLink("root_arm_1_link_2", noInertiaLink)

    x_joint = idyn.PrismaticJoint(
        world,
        root_arm_1_link_1,
        idyn.Transform(idyn.Rotation.Identity(), idyn.Position.Zero()),
        idyn.Axis(idyn.Direction(1.0, 0.0, 0.0), idyn.Position.Zero()),
    )
    x_joint.setPosLimits(0, -20, 20)
    x_joint.enablePosLimits(True)
    model.addJoint(
        "root_x_axis_joint",
        x_joint,
    )

    y_joint = idyn.PrismaticJoint(
        root_arm_1_link_1,
        root_arm_1_link_2,
        idyn.Transform(idyn.Rotation.Identity(), idyn.Position.Zero()),
        idyn.Axis(idyn.Direction(0.0, 1.0, 0.0), idyn.Position.Zero()),
    )
    y_joint.setPosLimits(0, -20, 20)
    y_joint.enablePosLimits(True)
    model.addJoint(
        "root_y_axis_joint",
        y_joint,
    )

    continuous_joint = idyn.RevoluteJoint(
        root_arm_1_link_2,
        root_link,
        idyn.Transform(idyn.Rotation.Identity(), idyn.Position(0, 0, 1)),
        idyn.Axis(idyn.Direction(0.0, 0.0, 1.0), idyn.Position.Zero()),
    )
    continuous_joint.enablePosLimits(False)
    model.addJoint("root_z_rotation_joint", continuous_joint)

    model.setDefaultBaseLink(world)

    urdf_dir = os.path.dirname(urdf_path_original)
    temp = tempfile.NamedTemporaryFile(dir=urdf_dir)

    # save the simplified model
    model_saver = idyn.ModelExporter()
    model_saver.init(model)
    model_saver.exportModelToFile(temp.name)

    urdf_path = temp.name

    # fix_root_link = False
    disable_self_collisions = True

    # Categorize joints by body part
    lower_body_joints = [
        "l_hip_pitch",
        "r_hip_pitch",
        "l_hip_roll",
        "r_hip_roll",
        "l_hip_yaw",
        "r_hip_yaw",
        "l_knee",
        "r_knee",
        "l_ankle_pitch",
        "r_ankle_pitch",
        "l_ankle_roll",
        "r_ankle_roll",
    ]

    upper_body_joints = [
        "torso_roll",
        "torso_pitch",
        "torso_yaw",
        "neck_pitch",
        "neck_roll",
        "neck_yaw",
        "camera_tilt",
    ]

    l_arm_joints = [
        "l_shoulder_pitch",
        "l_shoulder_roll",
        "l_shoulder_yaw",
        "l_elbow",
        "l_wrist_yaw",
        "l_wrist_roll",
        "l_wrist_pitch",
    ]

    r_arm_joints = [
        "r_shoulder_pitch",
        "r_shoulder_roll",
        "r_shoulder_yaw",
        "r_elbow",
        "r_wrist_yaw",
        "r_wrist_roll",
        "r_wrist_pitch",
    ]

    l_hand_joints = [
        "l_index_add",
        "l_middle_prox",
        "l_pinkie_prox",
        "l_ring_prox",
        "l_thumb_add",
        "l_index_prox",
        "l_middle_dist",
        "l_pinkie_dist",
        "l_ring_dist",
        "l_thumb_prox",
        "l_index_dist",
        "l_thumb_dist",
    ]

    r_hand_joints = [
        "r_index_add",
        "r_middle_prox",
        "r_pinkie_prox",
        "r_ring_prox",
        "r_thumb_add",
        "r_index_prox",
        "r_middle_dist",
        "r_pinkie_dist",
        "r_ring_dist",
        "r_thumb_prox",
        "r_index_dist",
        "r_thumb_dist",
    ]

    base_joint_names = [
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_rotation_joint",
    ]

    ## Controller parameters
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    hand_stiffness = 10
    hand_damping = 100
    hand_force_limit = 10

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 1]),
        ),
    )

    @property
    def _controller_configs(self):

        # -------------------------------------------------------------------------- #
        # Arms
        # -------------------------------------------------------------------------- #
        ## Joint Position Controllers
        l_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.l_arm_joints,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        r_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.r_arm_joints,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )

        ## EE Controllers
        l_arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.l_arm_joints,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link="l_hand_palm",
            urdf_path=self.urdf_path,
        )
        r_arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.r_arm_joints,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link="r_hand_palm",
            urdf_path=self.urdf_path,
        )

        # -------------------------------------------------------------------------- #
        # Hands
        # -------------------------------------------------------------------------- #
        ## Joint Position Controllers
        l_hand_pd_joint_pos = PDJointPosControllerConfig(
            self.l_hand_joints,
            None,
            None,
            self.hand_stiffness,
            self.hand_damping,
            self.hand_force_limit,
            normalize_action=False,
        )
        r_hand_pd_joint_pos = PDJointPosControllerConfig(
            self.r_hand_joints,
            None,
            None,
            self.hand_stiffness,
            self.hand_damping,
            self.hand_force_limit,
            normalize_action=False,
        )

        # -------------------------------------------------------------------------- #
        # Body
        # -------------------------------------------------------------------------- #
        ## Joint Position Controllers
        upper_body_pd_joint_pos = PDJointPosControllerConfig(
            self.upper_body_joints,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )

        # -------------------------------------------------------------------------- #
        # Base
        # -------------------------------------------------------------------------- #
        base_pd_joint_vel = PDBaseVelControllerConfig(
            self.base_joint_names,
            lower=[-1, -1, -3.14],
            upper=[1, 1, 3.14],
            damping=1000,
            force_limit=500,
        )

        controller_config = dict(
            pd_joint_pos=dict(
                l_arm=l_arm_pd_joint_pos,
                r_arm=r_arm_pd_joint_pos,
                l_hand=l_hand_pd_joint_pos,
                r_hand=r_hand_pd_joint_pos,
                upper_body=upper_body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pos=dict(
                l_arm=l_arm_pd_ee_delta_pos,
                r_arm=r_arm_pd_ee_delta_pos,
                l_hand=l_hand_pd_joint_pos,
                r_hand=r_hand_pd_joint_pos,
                upper_body=upper_body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
        )

        return deepcopy_dict(controller_config)

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="realsense",
                pose=sapien.Pose(p=[0, 0, 0.05], q=[1, 0, 0, 0]),
                width=640,
                height=480,
                fov=np.pi / 2,
                near=0.05,
                far=100,
                mount=self.robot.links_map["realsense"],
            )
        ]

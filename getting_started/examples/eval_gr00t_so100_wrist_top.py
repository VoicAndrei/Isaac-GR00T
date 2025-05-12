# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SO100 Real Robot with Wrist and Top Camera Support
import time
from contextlib import contextmanager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError

# Import the ExternalRobotInferenceClient from the correct path
from gr00t.eval.service import ExternalRobotInferenceClient

# Import tqdm for progress bar
from tqdm import tqdm

#################################################################################


class SO100WristTopRobot:
    def __init__(self, calibrate=False, enable_camera=False, camera_top_index=0, camera_wrist_index=4):
        self.config = So100RobotConfig()
        self.calibrate = calibrate
        self.enable_camera = enable_camera
        self.camera_top_index = camera_top_index
        self.camera_wrist_index = camera_wrist_index
        
        if not enable_camera:
            self.config.cameras = {}
        else:
            self.config.cameras = {
                "top": OpenCVCameraConfig(camera_top_index, 30, 640, 480, "bgr"),
                "wrist": OpenCVCameraConfig(camera_wrist_index, 30, 640, 480, "bgr")
            }
        self.config.leader_arms = {}

        # remove the .cache/calibration/so100 folder
        if self.calibrate:
            import os
            import shutil

            calibration_folder = os.path.join(os.getcwd(), ".cache", "calibration", "so100")
            print("========> Deleting calibration_folder:", calibration_folder)
            if os.path.exists(calibration_folder):
                shutil.rmtree(calibration_folder)

        # Create the robot
        self.robot = make_robot_from_config(self.config)
        self.motor_bus = self.robot.follower_arms["main"]

    @contextmanager
    def activate(self):
        try:
            self.connect()
            self.move_to_initial_pose()
            yield
        finally:
            self.disconnect()

    def connect(self):
        if self.robot.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        # Connect the arms
        self.motor_bus.connect()

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

        # Calibrate the robot
        self.robot.activate_calibration()

        self.set_so100_robot_preset()

        # Enable torque on all motors of the follower arms
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)
        print("robot present position:", self.motor_bus.read("Present_Position"))
        self.robot.is_connected = True

        # Connect cameras
        self.top_camera = self.robot.cameras.get("top") if self.enable_camera else None
        self.wrist_camera = self.robot.cameras.get("wrist") if self.enable_camera else None
        
        if self.top_camera is not None:
            self.top_camera.connect()
        if self.wrist_camera is not None:
            self.wrist_camera.connect()
            
        print("================> SO100 Robot with wrist and top cameras is fully connected =================")

    def set_so100_robot_preset(self):
        # Mode=0 for Position Control
        self.motor_bus.write("Mode", 0)
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        # self.motor_bus.write("P_Coefficient", 16)
        self.motor_bus.write("P_Coefficient", 10)
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        self.motor_bus.write("I_Coefficient", 0)
        self.motor_bus.write("D_Coefficient", 32)
        # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
        # which is mandatory for Maximum_Acceleration to take effect after rebooting.
        self.motor_bus.write("Lock", 0)
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        self.motor_bus.write("Maximum_Acceleration", 254)
        self.motor_bus.write("Acceleration", 254)

    def move_to_initial_pose(self):
        current_state = self.robot.capture_observation()["observation.state"]
        print("current_state", current_state)
        # print all keys of the observation
        print("observation keys:", self.robot.capture_observation().keys())

        current_state[0] = 90
        current_state[2] = 90
        current_state[3] = 90
        self.robot.send_action(current_state)
        time.sleep(2)

        current_state[4] = -70
        current_state[5] = 30
        current_state[1] = 90
        self.robot.send_action(current_state)
        time.sleep(2)

        print("----------------> SO100 Robot moved to initial pose")

    def go_home(self):
        # [ 88.0664, 156.7090, 135.6152,  83.7598, -89.1211,  16.5107]
        print("----------------> SO100 Robot moved to home pose")
        home_state = torch.tensor([88.0664, 156.7090, 135.6152, 83.7598, -89.1211, 16.5107])
        self.set_target_state(home_state)
        time.sleep(2)

    def get_observation(self):
        return self.robot.capture_observation()

    def get_current_state(self):
        return self.get_observation()["observation.state"].data.numpy()

    def get_current_images(self):
        """Get images from both cameras"""
        observation = self.get_observation()
        images = {}
        
        if "observation.images.top" in observation:
            top_img = observation["observation.images.top"].data.numpy()
            # convert bgr to rgb
            images["top"] = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)
            
        if "observation.images.wrist" in observation:
            wrist_img = observation["observation.images.wrist"].data.numpy()
            # convert bgr to rgb
            images["wrist"] = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)
            
        return images

    def set_target_state(self, target_state: torch.Tensor):
        self.robot.send_action(target_state)

    def enable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)

    def disable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

    def disconnect(self):
        self.disable()
        self.robot.disconnect()
        self.robot.is_connected = False
        print("================> SO100 Robot disconnected")

    def __del__(self):
        self.disconnect()


#################################################################################


class Gr00tWristTopInferenceClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        language_instruction="Pick up the green cubes from the white plate and place them in the black recipient",
    ):
        self.language_instruction = language_instruction
        # 480, 640
        self.img_size = (480, 640)
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, images, state):
        """
        Get action from policy using images from both cameras
        
        Args:
            images: Dictionary with 'top' and 'wrist' camera images
            state: Robot state
        """
        obs_dict = {
            "video.top": images["top"][np.newaxis, :, :, :],
            "video.wrist": images["wrist"][np.newaxis, :, :, :],
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "annotation.human.task_description": [self.language_instruction],
        }
        start_time = time.time()
        res = self.policy.get_action(obs_dict)
        print("Inference query time taken", time.time() - start_time)
        return res

    def sample_action(self):
        obs_dict = {
            "video.top": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "video.wrist": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "state.single_arm": np.zeros((1, 5)),
            "state.gripper": np.zeros((1, 1)),
            "annotation.human.task_description": [self.language_instruction],
        }
        return self.policy.get_action(obs_dict)


#################################################################################


def view_multiple_cameras(images):
    """
    Display multiple camera views side by side without creating new windows each time
    """
    # Use a single figure that's reused
    plt.figure(1, figsize=(12, 5))
    plt.clf()  # Clear the current figure
    
    num_cameras = len(images)
    for i, (camera_name, img) in enumerate(images.items()):
        plt.subplot(1, num_cameras, i+1)
        plt.imshow(img)
        plt.title(camera_name)
        plt.axis("off")
        
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)  # Non-blocking show


#################################################################################

if __name__ == "__main__":
    import argparse
    import os

    default_dataset_path = os.path.expanduser("~/datasets/so100_strawberry_grape")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_policy", action="store_true"
    )  # default is to playback the provided dataset
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--action_horizon", type=int, default=12)
    parser.add_argument("--actions_to_execute", type=int, default=300)
    parser.add_argument("--camera_top_index", type=int, default=0)
    parser.add_argument("--camera_wrist_index", type=int, default=4)
    args = parser.parse_args()

    ACTIONS_TO_EXECUTE = args.actions_to_execute
    USE_POLICY = args.use_policy
    ACTION_HORIZON = args.action_horizon  # we will execute only some actions from the action_chunk of 16
    MODALITY_KEYS = ["single_arm", "gripper"]

    if USE_POLICY:
        client = Gr00tWristTopInferenceClient(
            host=args.host,
            port=args.port,
            language_instruction="Pick up the green cubes from the white plate and place them in the black recipient",
        )

        robot = SO100WristTopRobot(
            calibrate=False, 
            enable_camera=True, 
            camera_top_index=args.camera_top_index,
            camera_wrist_index=args.camera_wrist_index
        )
        
        with robot.activate():
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):
                images = robot.get_current_images()
                view_multiple_cameras(images)
                
                state = robot.get_current_state()
                action = client.get_action(images, state)
                
                start_time = time.time()
                for i in range(ACTION_HORIZON):
                    concat_action = np.concatenate(
                        [np.atleast_1d(action[f"action.{key}"][i]) for key in MODALITY_KEYS],
                        axis=0,
                    )
                    assert concat_action.shape == (6,), concat_action.shape
                    robot.set_target_state(torch.from_numpy(concat_action))
                    time.sleep(0.02)

                    # get the realtime images
                    images = robot.get_current_images()
                    view_multiple_cameras(images)

                    # 0.05*16 = 0.8 seconds
                    print("executing action", i, "time taken", time.time() - start_time)
                print("Action chunk execution time taken", time.time() - start_time)
    else:
        print("Playback from dataset not supported in wrist-top camera mode. Please use --use_policy") 
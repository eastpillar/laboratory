"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.
Modified to support ECoT (Embodied Chain-of-Thought) visualization.
"""

import os
import sys
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, List

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.bridge.utils import get_action
from experiments.robot.robot_utils import (
    DATE_TIME,
    # get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

# [ECoT] Bridge Utils에서 시각화 함수 임포트
# (경로 문제 발생 시 sys.path 설정 확인 필요)
from experiments.bridge.utils import (
    draw_bboxes,
    draw_gripper,
    make_reasoning_image,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"  # Model family
    pretrained_checkpoint: Union[str, Path] = ""  # Pretrained checkpoint path
    load_in_8bit: bool = False  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True  # Center crop? (if trained w/ random crop image aug)

    use_ecot: bool = True  # Enable Embodied Chain-of-Thought reasoning & visualization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None  # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"  # Local directory for eval logs

    use_wandb: bool = False  # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"  # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"  # Name of entity to log under

    seed: int = 7  # Random Seed (for reproducibility)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    set_seed_everywhere(cfg.seed)

    cfg.unnorm_key = cfg.task_suite_name

    model = get_model(cfg)
    if cfg.model_family == "openvla":
        if cfg.unnorm_key in model.norm_stats:
            pass
        elif "bridge_reasoning" in model.norm_stats:
            print(f"Warning: Key '{cfg.unnorm_key}' not found. Using 'bridge_reasoning' stats for ECoT model.")
            cfg.unnorm_key = "bridge_reasoning"
        elif "bridge_orig" in model.norm_stats:
            print(f"Warning: Key '{cfg.unnorm_key}' not found. Using 'bridge_orig' stats.")
            cfg.unnorm_key = "bridge_orig"
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Final Error: Key {cfg.unnorm_key} not found in {list(model.norm_stats.keys())}"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.use_ecot:
        run_id += "-ECoT"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg) # 224
    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400
            else:
                max_steps = 300

            print(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")

            current_reasoning_cache = None
            REASONING_FREQ = 3  # 5번마다 생각함
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)
                    # Prepare observations dict
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    # [ECoT] Create info_dict to capture reasoning
                    info_dict = dict()
                    should_reason = (t % REASONING_FREQ == 0) or (current_reasoning_cache is None)
                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        info_dict=info_dict,  # [ECoT] Pass dictionary to receive tokens
                        unnorm_key=cfg.unnorm_key,
                        max_new_tokens=1024,
                        use_reasoning=should_reason,
                        cached_reasoning=current_reasoning_cache
                    )
                    if should_reason and "decoded_tokens" in info_dict:
                        current_reasoning_cache = info_dict["decoded_tokens"]
                    # [ECoT] Visualization Logic
                    video_image = img.copy()
                    if cfg.use_ecot:
                        try:
                            reasoning_img, metadata = make_reasoning_image(info_dict["decoded_tokens"])
                            draw_gripper(video_image, metadata["gripper"])
                            draw_bboxes(video_image, metadata["bboxes"])

                            video_image = np.concatenate([video_image, reasoning_img], axis=1)
                        except Exception as e:
                            video_image = img

                    replay_images.append(video_image)
                    cv2.imshow('asd',video_image)
                    cv2.waitKey(1)
                    # Normalize gripper action [0,1] -> [-1,+1]
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] Flip gripper action sign
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    import traceback
                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
"""
Relabel actions in LIBERO processed datasets using the pi-0.5 LIBERO policy.

For every observation in the source dataset, we run the policy in direct
inference mode and store the returned action chunk (T, D) alongside the
original trajectory data in a new HDF5 file.
"""

import dataclasses
import json
import logging
import pathlib
from typing import Optional

import h5py
import numpy as np
import tqdm
import tyro

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config


def load_policy():
    """Load the pi-0.5 LIBERO policy using direct inference (not server)."""
    logging.info("Loading pi-0.5 LIBERO policy...")
    config = _config.get_config("pi05_libero")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    logging.info("✓ Policy loaded")
    return policy


def build_policy_observation(
    agent_img: np.ndarray,
    wrist_img: np.ndarray,
    ee_pos: np.ndarray,
    ee_ori_axis_angle: np.ndarray,
    gripper_states: np.ndarray,
    prompt: str,
    resize: int,
):
    """Convert stored LIBERO observations to the policy input format."""
    from openpi_client import image_tools

    # Rotate 180 degrees to match policy training convention, then resize.
    agent_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(agent_img[::-1, ::-1], resize, resize)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img[::-1, ::-1], resize, resize)
    )

    obs_state = np.concatenate([ee_pos, ee_ori_axis_angle, gripper_states])
    return {
        "observation/image": agent_img,
        "observation/wrist_image": wrist_img,
        "observation/state": obs_state,
        "prompt": str(prompt),
    }


@dataclasses.dataclass
class Args:
    # Dataset configuration
    task_suite_name: str = "libero_spatial"
    dataset_root: str = "libero_openvla_processed_hdf5"
    output_root: str = "libero_openvla_processed_hdf5_pi05_actions"

    # Limits for quick debugging
    max_demos: Optional[int] = None
    max_steps: Optional[int] = None

    # Pre-processing
    resize: int = 224
    overwrite: bool = False


def relabel_file(h5_path: pathlib.Path, output_path: pathlib.Path, policy, args: Args):
    logging.info(f"Relabeling {h5_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        logging.info(f"Skipping existing file (overwrite=False): {output_path}")
        return

    with h5py.File(h5_path, "r") as src_file, h5py.File(output_path, "w") as dst_file:
        # Copy the full data group first so all original content is preserved.
        src_file.copy("data", dst_file)

        src_data_group = src_file["data"]
        dst_data_group = dst_file["data"]

        problem_info = json.loads(src_data_group.attrs["problem_info"])
        prompt = problem_info.get("language_instruction", "")

        demo_names = sorted(list(src_data_group.keys()))
        if args.max_demos is not None:
            demo_names = demo_names[: args.max_demos]

        for demo_name in tqdm.tqdm(demo_names, desc=h5_path.name, leave=False):
            src_demo = src_data_group[demo_name]
            dst_demo = dst_data_group[demo_name]
            obs = src_demo["obs"]

            num_steps = obs["agentview_rgb"].shape[0]
            if args.max_steps is not None:
                num_steps = min(num_steps, args.max_steps)

            action_chunks = None
            chunk_len = None
            action_dim = None

            for t in range(num_steps):
                obs_dict = build_policy_observation(
                    agent_img=obs["agentview_rgb"][t],
                    wrist_img=obs["eye_in_hand_rgb"][t],
                    ee_pos=obs["ee_pos"][t],
                    ee_ori_axis_angle=obs["ee_ori"][t],
                    gripper_states=obs["gripper_states"][t],
                    prompt=prompt,
                    resize=args.resize,
                )

                result = policy.infer(obs_dict)
                actions = np.asarray(result["actions"])

                if action_chunks is None:
                    chunk_len, action_dim = actions.shape
                    action_chunks = np.zeros(
                        (num_steps, chunk_len, action_dim), dtype=np.float32
                    )
                elif actions.shape != (chunk_len, action_dim):
                    raise ValueError(
                        f"Inconsistent action chunk shape for {demo_name} "
                        f"step {t}: expected {(chunk_len, action_dim)}, got {actions.shape}"
                    )

                action_chunks[t] = actions.astype(np.float32)

            dst_demo.create_dataset("pi05_action_chunks", data=action_chunks, compression="gzip")
            dst_demo.attrs["pi05_chunk_length"] = chunk_len
            dst_demo.attrs["pi05_action_dim"] = action_dim
            dst_demo.attrs["pi05_prompt"] = prompt
            dst_demo.attrs["pi05_policy"] = "pi05_libero"

        # If demos were truncated for debugging, keep metadata consistent.
        dst_data_group.attrs["num_demos"] = len(dst_data_group.keys())
        dst_data_group.attrs["total"] = sum(
            dst_data_group[demo].attrs["num_samples"] for demo in dst_data_group.keys()
        )


def main(args: Args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    input_dir = pathlib.Path(args.dataset_root) / f"{args.task_suite_name}_openvla_processed"
    output_dir = pathlib.Path(args.output_root) / f"{args.task_suite_name}_openvla_processed"

    if not input_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {input_dir}")

    policy = load_policy()

    h5_paths = sorted(input_dir.glob("*.hdf5"))
    if not h5_paths:
        raise FileNotFoundError(f"No .hdf5 files found under {input_dir}")

    for h5_path in tqdm.tqdm(h5_paths, desc="Datasets"):
        output_path = output_dir / h5_path.name
        relabel_file(h5_path, output_path, policy, args)


if __name__ == "__main__":
    tyro.cli(main)

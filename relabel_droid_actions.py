"""
Relabel DROID RLDS datasets with pi-0.5 DROID policy actions.

For every step in the source dataset, the policy is run in direct inference
mode and the returned action chunk (T, D) is stored in an LMDB database keyed
by the unique step ID defined in droid_rlds_dataset.py:

    key = "<recording_folderpath>--<file_path>--<step_index>"

This key matches the ``step_id`` field produced by DroidRldsDataset, so the
relabeled actions can be looked up efficiently during training.

Storage layout (LMDB)
---------------------
key   : step_id as UTF-8 bytes
value : 8-byte little-endian header (uint32 chunk_len, uint32 action_dim)
        followed by a flat float32 array of shape (chunk_len * action_dim)

The stored action chunk corresponds to policy inference using
``exterior_image_1_left`` as the base camera.  To keep the training data
consistent, the ``droid_rlds_dataset.py`` restructure function should be
updated to always use ``exterior_image_1_left`` (rather than randomly picking
between cam 1 and cam 2) when relabeled actions are loaded.

Multi-GPU / distributed usage
------------------------------
Launch one process per shard with ``--shard-index`` and ``--num-shards``:

    CUDA_VISIBLE_DEVICES=0 python relabel_droid_actions.py \\
        --data-dir /data/droid --shard-index 0 --num-shards 8

    CUDA_VISIBLE_DEVICES=1 python relabel_droid_actions.py \\
        --data-dir /data/droid --shard-index 1 --num-shards 8

    ... (repeat for all shards) ...

Then merge the shard databases into a single LMDB:

    python relabel_droid_actions.py --merge \\
        --output-dir /data/droid_relabeled

Loading during training
-----------------------
    import lmdb, struct, numpy as np

    env = lmdb.open(output_dir, readonly=True, lock=False)
    txn = env.begin()

    raw = txn.get(step_id.encode())   # step_id is a Python str
    chunk_len, action_dim = struct.unpack_from("<II", raw, 0)
    actions = np.frombuffer(raw, dtype=np.float32, offset=8).reshape(chunk_len, action_dim)
"""

import dataclasses
import logging
import pathlib
import struct
from typing import Optional

import lmdb
import numpy as np
import tqdm
import tyro

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config


# ---------------------------------------------------------------------------
# LMDB helpers
# ---------------------------------------------------------------------------

LMDB_MAP_SIZE = 500 * 1024**3  # 500 GB virtual address space (sparse on Linux)
LMDB_COMMIT_EVERY = 2_000       # write transaction commit interval (in steps)


def encode_actions(actions: np.ndarray) -> bytes:
    """Pack an (chunk_len, action_dim) float32 array into bytes."""
    chunk_len, action_dim = actions.shape
    header = struct.pack("<II", chunk_len, action_dim)
    return header + actions.astype(np.float32).tobytes()


def decode_actions(value: bytes) -> np.ndarray:
    """Unpack bytes produced by encode_actions back to an ndarray."""
    chunk_len, action_dim = struct.unpack_from("<II", value, 0)
    return np.frombuffer(value, dtype=np.float32, offset=8).reshape(chunk_len, action_dim)


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_policy(config_name: str, checkpoint_path: str):
    logging.info(f"Loading policy '{config_name}' from '{checkpoint_path}' ...")
    config = _config.get_config(config_name)
    checkpoint_dir = download.maybe_download(checkpoint_path)
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    logging.info("Policy loaded.")
    return policy


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def build_policy_observation(
    exterior_img: np.ndarray,
    wrist_img: np.ndarray,
    joint_position: np.ndarray,
    gripper_position: np.ndarray,
    prompt: str,
    resize: int,
) -> dict:
    """Convert raw DROID observations to the pi05-droid policy input format.

    Input images may be any uint8 HWC array; they are resized to (resize, resize).
    The key names match ``make_droid_example()`` in droid_policy.py.
    """
    from openpi_client import image_tools

    exterior_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(exterior_img, resize, resize)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize, resize)
    )

    gripper = np.asarray(gripper_position).reshape(-1)  # ensure 1-D

    return {
        "observation/exterior_image_1_left": exterior_img,
        "observation/wrist_image_left": wrist_img,
        "observation/joint_position": np.asarray(joint_position),
        "observation/gripper_position": gripper,
        "prompt": str(prompt),
    }


# ---------------------------------------------------------------------------
# Per-dataset relabeling
# ---------------------------------------------------------------------------

def relabel_dataset(
    ds_name: str,
    ds_version: str,
    data_dir: str,
    lmdb_env: lmdb.Environment,
    policy,
    *,
    resize: int,
    shard_index: int,
    num_shards: int,
    max_episodes: Optional[int],
    max_steps_per_episode: Optional[int],
) -> int:
    """Iterate over one RLDS dataset and write relabeled actions to lmdb_env.

    Returns the total number of steps written.
    """
    import dlimp as dl
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")

    builder = tfds.builder(ds_name, data_dir=data_dir, version=ds_version)
    dataset = dl.DLataset.from_rlds(
        builder, split="train", shuffle=False, num_parallel_reads=1
    )

    # Keep only successful trajectories.
    dataset = dataset.filter(
        lambda traj: tf.strings.regex_full_match(
            traj["traj_metadata"]["episode_metadata"]["file_path"][0],
            ".*success.*",
        )
    )

    # Assign episodes to shards deterministically.
    if num_shards > 1:
        dataset = dataset.shard(num_shards, shard_index)

    if max_episodes is not None:
        dataset = dataset.take(max_episodes)

    steps_written = 0
    pending = 0  # steps written since last commit

    txn = lmdb_env.begin(write=True)

    for traj in tqdm.tqdm(dataset.as_numpy_iterator(), desc=f"{ds_name}:{ds_version}", unit="ep"):
        # ---- Extract trajectory-level metadata ----
        ep_meta = traj["traj_metadata"]["episode_metadata"]
        # Both fields are per-step tensors but constant within an episode.
        recording_folder = ep_meta["recording_folderpath"][0].decode()
        file_path = ep_meta["file_path"][0].decode()

        # Use the first language instruction (consistent with make_droid_example).
        prompt = traj["language_instruction"].decode()

        # ---- Per-step arrays ----
        ext_imgs_1 = traj["observation"]["exterior_image_1_left"]   # (T,) encoded
        wrist_imgs = traj["observation"]["wrist_image_left"]         # (T,) encoded
        joint_pos = traj["observation"]["joint_position"]            # (T, 7)
        gripper_pos = traj["observation"]["gripper_position"]        # (T, 1)

        num_steps = joint_pos.shape[0]
        if max_steps_per_episode is not None:
            num_steps = min(num_steps, max_steps_per_episode)

        for t in range(num_steps):
            # Decode JPEG-encoded images to uint8 HWC numpy arrays.
            ext_img = tf.io.decode_image(
                ext_imgs_1[t], expand_animations=False, dtype=tf.uint8
            ).numpy()
            wrist_img = tf.io.decode_image(
                wrist_imgs[t], expand_animations=False, dtype=tf.uint8
            ).numpy()

            obs = build_policy_observation(
                exterior_img=ext_img,
                wrist_img=wrist_img,
                joint_position=joint_pos[t],
                gripper_position=gripper_pos[t],
                prompt=prompt,
                resize=resize,
            )

            result = policy.infer(obs)
            actions = np.asarray(result["actions"], dtype=np.float32)

            # Construct the same step_id as droid_rlds_dataset.py.
            step_id = f"{recording_folder}--{file_path}--{t}"
            txn.put(step_id.encode(), encode_actions(actions))

            steps_written += 1
            pending += 1

            if pending >= LMDB_COMMIT_EVERY:
                txn.commit()
                txn = lmdb_env.begin(write=True)
                pending = 0

    txn.commit()
    return steps_written


# ---------------------------------------------------------------------------
# Merge shards
# ---------------------------------------------------------------------------

def merge_shards(output_dir: pathlib.Path, num_shards: int) -> None:
    """Combine all per-shard LMDB databases into a single merged database."""
    merged_path = output_dir / "relabeled_actions.lmdb"
    logging.info(f"Merging {num_shards} shards into {merged_path} ...")

    shard_paths = []
    for i in range(num_shards):
        p = output_dir / f"shard_{i:05d}_of_{num_shards:05d}.lmdb"
        if not p.exists():
            raise FileNotFoundError(f"Shard not found: {p}")
        shard_paths.append(p)

    with lmdb.open(str(merged_path), map_size=LMDB_MAP_SIZE) as merged_env:
        for shard_path in tqdm.tqdm(shard_paths, desc="Merging shards"):
            with lmdb.open(str(shard_path), readonly=True, lock=False) as shard_env:
                with shard_env.begin() as src_txn, merged_env.begin(write=True) as dst_txn:
                    cursor = src_txn.cursor()
                    for key, value in cursor.iternext_dup() if False else cursor.iternext(keys=True, values=True):
                        dst_txn.put(key, value)

    logging.info(f"Merge complete: {merged_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    # Data
    data_dir: str = "/data/droid"
    """Root directory of the DROID RLDS dataset(s)."""

    datasets: list[str] = dataclasses.field(
        default_factory=lambda: ["droid:1.0.0"]
    )
    """Datasets to process, each as '<name>:<version>'."""

    output_dir: str = "droid_relabeled_actions"
    """Directory for LMDB output file(s)."""

    # Policy
    policy_config_name: str = "pi05_droid"
    """openpi training config name for the policy."""

    checkpoint_path: str = "gs://openpi-assets/checkpoints/pi05_droid"
    """GCS or local path to the policy checkpoint."""

    resize: int = 224
    """Target image size (square) for policy inference."""

    # Sharding (for multi-GPU parallel relabeling)
    shard_index: int = 0
    """Index of this shard (0-based)."""

    num_shards: int = 1
    """Total number of shards. Set >1 for multi-GPU parallel relabeling."""

    # Merge mode
    merge: bool = False
    """If True, merge all existing shard LMDBs and exit (no inference)."""

    # Debugging limits
    max_episodes: Optional[int] = None
    max_steps_per_episode: Optional[int] = None


def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Merge mode ----
    if args.merge:
        # Discover how many shards exist.
        shard_files = sorted(output_dir.glob("shard_*_of_*.lmdb"))
        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {output_dir}")
        # Parse num_shards from filename.
        num_shards = int(shard_files[0].stem.split("_of_")[1])
        merge_shards(output_dir, num_shards)
        return

    # ---- Relabeling mode ----
    if args.num_shards > 1:
        lmdb_path = output_dir / f"shard_{args.shard_index:05d}_of_{args.num_shards:05d}.lmdb"
    else:
        lmdb_path = output_dir / "relabeled_actions.lmdb"

    logging.info(f"Output LMDB: {lmdb_path}")

    policy = load_policy(args.policy_config_name, args.checkpoint_path)

    total_steps = 0
    with lmdb.open(str(lmdb_path), map_size=LMDB_MAP_SIZE) as lmdb_env:
        for ds_spec in args.datasets:
            if ":" not in ds_spec:
                raise ValueError(f"Dataset spec must be '<name>:<version>', got: {ds_spec!r}")
            ds_name, ds_version = ds_spec.split(":", 1)
            steps = relabel_dataset(
                ds_name=ds_name,
                ds_version=ds_version,
                data_dir=args.data_dir,
                lmdb_env=lmdb_env,
                policy=policy,
                resize=args.resize,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                max_episodes=args.max_episodes,
                max_steps_per_episode=args.max_steps_per_episode,
            )
            logging.info(f"  {ds_name}:{ds_version} → {steps:,} steps written")
            total_steps += steps

    logging.info(f"Done. Total steps written: {total_steps:,}")
    logging.info(f"LMDB written to: {lmdb_path}")


if __name__ == "__main__":
    tyro.cli(main)

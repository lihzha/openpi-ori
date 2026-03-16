"""
Relabel DROID RLDS datasets with pi-0.5 DROID policy actions.

For every step in the source dataset, the policy is run in direct inference
mode and the returned action chunk (T, D) is stored in a single ArrayRecord
file, with a companion JSON index that maps each step_id to its integer row:

    key  : "<recording_folderpath>--<file_path>--<step_index>"
    value: row index (int) into relabeled_actions.arrayrecord

This key matches the ``step_id`` field produced by DroidRldsDataset.

Output files
------------
relabeled_actions.arrayrecord  – all action chunks, one record per row
relabeled_actions_index.json   – {"<step_id>": <row_index>, ...}

Each record is the raw bytes produced by encode_actions():
    8-byte little-endian header (uint32 chunk_len, uint32 action_dim)
    followed by a flat float32 array of shape (chunk_len * action_dim)

Single-machine / multi-process TPU usage
-----------------------------------------
On a single TPU VM all devices are visible to one process by default:

    python relabel_droid_actions.py --data-dir /data/droid

For explicit multi-process execution (one process per chip), set the standard
JAX distributed environment variables before launching:

    JAX_COORDINATOR_ADDRESS=localhost:1234 JAX_NUM_PROCESSES=4 JAX_PROCESS_ID=0 \\
        python relabel_droid_actions.py --data-dir /data/droid &
    JAX_COORDINATOR_ADDRESS=localhost:1234 JAX_NUM_PROCESSES=4 JAX_PROCESS_ID=1 \\
        python relabel_droid_actions.py --data-dir /data/droid &
    ... (repeat for all processes) ...

Each process writes its own per-process shard. Process 0 then merges all
shards into a single ArrayRecord file and uploads both output files to GCS
(if output_dir is a gs:// path).

GCS output
----------
Shards are written to a local temporary directory. After all processes
finish, process 0 merges them into one ArrayRecord + one index JSON and
uploads both to GCS via tf.io.gfile.

Loading during training
-----------------------
    import json, struct
    import numpy as np
    import tensorflow as tf
    from array_record.python.array_record_module import ArrayRecordReader

    with tf.io.gfile.GFile("gs://.../relabeled_actions_index.json") as f:
        index = json.load(f)                           # {step_id: row_int}

    reader = ArrayRecordReader("gs://.../relabeled_actions.arrayrecord")

    def lookup(step_id: str) -> np.ndarray:
        raw = reader.read([index[step_id]])[0]
        chunk_len, action_dim = struct.unpack_from("<II", raw, 0)
        return np.frombuffer(raw, dtype=np.float32, offset=8).reshape(chunk_len, action_dim)
"""

import dataclasses
import json
import logging
import os
import pathlib
import struct
import tempfile
from typing import Optional

import jax
import jax.experimental.multihost_utils as mhu
import numpy as np
import tqdm
import tyro
from array_record.python.array_record_module import ArrayRecordReader, ArrayRecordWriter

from openpi.policies import policy_config
from openpi.training import config as _config


# One record per riegeli chunk → O(1) random access with minimal read
# amplification; ideal for lookup-by-index during training.
ARRAYRECORD_OPTIONS = "group_size:1"

# Number of records to read at once when merging shards.
MERGE_BATCH = 10_000


# ---------------------------------------------------------------------------
# Record encoding
# ---------------------------------------------------------------------------

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
    policy = policy_config.create_trained_policy(config, checkpoint_path)
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
    writer: ArrayRecordWriter,
    index: dict,
    policy,
    *,
    resize: int,
    max_episodes: Optional[int],
    max_steps_per_episode: Optional[int],
) -> int:
    """Iterate over one RLDS dataset and append relabeled actions to writer.

    index is mutated in-place: index[step_id] = row_int (next free row).

    Data is sharded across JAX processes via jax.process_count() /
    jax.process_index(), consistent with the rest of the training codebase.

    Returns the total number of steps written by this process.
    """
    import dlimp as dl
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")

    proc_idx = jax.process_index()
    num_procs = jax.process_count()

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

    # Shard across JAX processes, consistent with mixins.py pattern.
    if num_procs > 1:
        dataset = dataset.shard(num_procs, proc_idx)

    if max_episodes is not None:
        dataset = dataset.take(max_episodes)

    steps_written = 0

    for traj in tqdm.tqdm(
        dataset.as_numpy_iterator(),
        desc=f"[proc {proc_idx}/{num_procs}] {ds_name}:{ds_version}",
        unit="ep",
    ):
        # ---- Extract trajectory-level metadata ----
        ep_meta = traj["traj_metadata"]["episode_metadata"]
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
            index[step_id] = len(index)
            writer.write(encode_actions(actions))

            steps_written += 1

    return steps_written


# ---------------------------------------------------------------------------
# Merge shards
# ---------------------------------------------------------------------------

def merge_shards(
    local_dir: pathlib.Path, num_shards: int
) -> tuple[pathlib.Path, pathlib.Path]:
    """Merge all per-process ArrayRecord shards into one file + one index.

    Reads shards in batches of MERGE_BATCH to avoid per-record Python overhead.
    Returns (arrayrecord_path, index_json_path).
    """
    merged_ar_path = local_dir / "relabeled_actions.arrayrecord"
    merged_idx_path = local_dir / "relabeled_actions_index.json"
    logging.info(f"Merging {num_shards} shards into {merged_ar_path} ...")

    merged_index: dict[str, int] = {}
    global_idx = 0

    writer = ArrayRecordWriter(str(merged_ar_path), ARRAYRECORD_OPTIONS)
    try:
        for i in tqdm.tqdm(range(num_shards), desc="Merging shards", unit="shard"):
            shard_ar = local_dir / f"shard_{i:05d}_of_{num_shards:05d}.arrayrecord"
            shard_idx = local_dir / f"shard_{i:05d}_of_{num_shards:05d}_index.json"

            if not shard_ar.exists():
                raise FileNotFoundError(f"Shard not found: {shard_ar}")
            if not shard_idx.exists():
                raise FileNotFoundError(f"Shard index not found: {shard_idx}")

            with open(shard_idx) as f:
                local_index: dict[str, int] = json.load(f)

            # Reverse: local row → step_id (built once per shard).
            row_to_step = {v: k for k, v in local_index.items()}

            reader = ArrayRecordReader(str(shard_ar))
            num_records = reader.num_records()

            for start in range(0, num_records, MERGE_BATCH):
                end = min(start + MERGE_BATCH, num_records)
                batch = reader.read(list(range(start, end)))
                for local_row, record in zip(range(start, end), batch):
                    writer.write(record)
                    merged_index[row_to_step[local_row]] = global_idx
                    global_idx += 1

            reader.close()
    finally:
        writer.close()

    with open(merged_idx_path, "w") as f:
        json.dump(merged_index, f)

    logging.info(f"Merge complete: {global_idx:,} records -> {merged_ar_path}")
    return merged_ar_path, merged_idx_path


# ---------------------------------------------------------------------------
# GCS upload helper
# ---------------------------------------------------------------------------

def _upload_to_gcs(local_path: pathlib.Path, gcs_dir: str) -> None:
    """Copy a local file to GCS using tf.io.gfile (single Class A write op)."""
    import tensorflow as tf

    gcs_path = tf.io.gfile.join(gcs_dir, local_path.name)
    logging.info(f"Uploading {local_path} -> {gcs_path} ...")
    with open(local_path, "rb") as f_in, tf.io.gfile.GFile(gcs_path, "wb") as f_out:
        while chunk := f_in.read(64 * 1024 * 1024):  # 64 MB chunks
            f_out.write(chunk)
    logging.info("Upload complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    # Data
    data_dir: str = "/data/droid"
    """Root directory of the DROID RLDS dataset(s)."""

    datasets: list[str] = dataclasses.field(
        default_factory=lambda: ["droid:1.0.1"]
    )
    """Datasets to process, each as '<name>:<version>'."""

    output_dir: str = "droid_relabeled_actions"
    """Directory for output files. May be a gs:// path; shards are then written
    to a local temporary directory and uploaded to GCS after merging."""

    # Policy
    policy_config_name: str = "pi05_droid"
    """openpi training config name for the policy."""

    checkpoint_path: str = "gs://openpi-assets/checkpoints/pi05_droid"
    """GCS or local path to the policy checkpoint."""

    resize: int = 224
    """Target image size (square) for policy inference."""

    # Merge mode
    merge: bool = False
    """If True, merge existing per-process shards in output_dir and exit.
    Useful for re-running the merge step without re-running inference."""

    # Debugging limits
    max_episodes: Optional[int] = None
    max_steps_per_episode: Optional[int] = None


def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize JAX distributed if running in a multi-process context.
    # On a single TPU VM all devices are in one process; this is a no-op.
    if os.environ.get("JAX_COORDINATOR_ADDRESS") or os.environ.get("SLURM_NTASKS"):
        jax.distributed.initialize()

    proc_idx = jax.process_index()
    num_procs = jax.process_count()
    logging.info(
        f"JAX: process {proc_idx}/{num_procs}, "
        f"local devices: {jax.local_device_count()}, "
        f"total devices: {jax.device_count()}"
    )

    # ArrayRecord cannot stream writes to GCS; write shards locally and upload.
    is_gcs_output = args.output_dir.startswith("gs://")
    if is_gcs_output:
        local_dir = pathlib.Path(tempfile.mkdtemp(prefix="droid_relabel_"))
        logging.info(f"GCS output detected; writing shards locally to {local_dir}")
    else:
        local_dir = pathlib.Path(args.output_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # ---- Merge-only mode ----
    if args.merge:
        mhu.sync_global_devices("pre_merge")
        if proc_idx == 0:
            shard_files = sorted(local_dir.glob("shard_*_of_*.arrayrecord"))
            if not shard_files:
                raise FileNotFoundError(f"No shard files found in {local_dir}")
            num_shards = int(shard_files[0].stem.split("_of_")[1])
            merged_ar, merged_idx = merge_shards(local_dir, num_shards)
            if is_gcs_output:
                _upload_to_gcs(merged_ar, args.output_dir)
                _upload_to_gcs(merged_idx, args.output_dir)
        mhu.sync_global_devices("post_merge")
        return

    # ---- Relabeling mode ----
    if num_procs > 1:
        ar_path = local_dir / f"shard_{proc_idx:05d}_of_{num_procs:05d}.arrayrecord"
        idx_path = local_dir / f"shard_{proc_idx:05d}_of_{num_procs:05d}_index.json"
    else:
        ar_path = local_dir / "relabeled_actions.arrayrecord"
        idx_path = local_dir / "relabeled_actions_index.json"

    logging.info(f"[proc {proc_idx}] Output: {ar_path}")

    policy = load_policy(args.policy_config_name, args.checkpoint_path)

    index: dict[str, int] = {}
    total_steps = 0
    writer = ArrayRecordWriter(str(ar_path), ARRAYRECORD_OPTIONS)
    try:
        for ds_spec in args.datasets:
            if ":" not in ds_spec:
                raise ValueError(f"Dataset spec must be '<name>:<version>', got: {ds_spec!r}")
            ds_name, ds_version = ds_spec.split(":", 1)
            steps = relabel_dataset(
                ds_name=ds_name,
                ds_version=ds_version,
                data_dir=args.data_dir,
                writer=writer,
                index=index,
                policy=policy,
                resize=args.resize,
                max_episodes=args.max_episodes,
                max_steps_per_episode=args.max_steps_per_episode,
            )
            logging.info(f"[proc {proc_idx}] {ds_name}:{ds_version} -> {steps:,} steps written")
            total_steps += steps
    finally:
        writer.close()

    with open(idx_path, "w") as f:
        json.dump(index, f)

    logging.info(f"[proc {proc_idx}] Done. Total steps: {total_steps:,}")

    # Wait for all processes to finish writing their shards.
    mhu.sync_global_devices("relabeling_complete")

    # Process 0 merges and (optionally) uploads to GCS.
    if proc_idx == 0:
        if num_procs > 1:
            merged_ar, merged_idx = merge_shards(local_dir, num_procs)
        else:
            merged_ar, merged_idx = ar_path, idx_path
        if is_gcs_output:
            _upload_to_gcs(merged_ar, args.output_dir)
            _upload_to_gcs(merged_idx, args.output_dir)

    mhu.sync_global_devices("merge_complete")
    logging.info(f"[proc {proc_idx}] All done.")


if __name__ == "__main__":
    tyro.cli(main)

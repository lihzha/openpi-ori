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

import flax.nnx as nnx

from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import sharding as _sharding


# One record per riegeli chunk → O(1) random access with minimal read
# amplification; ideal for lookup-by-index during training.
ARRAYRECORD_OPTIONS = "group_size:1"

# Number of records to read at once when merging shards.
MERGE_BATCH = 10_000


# ---------------------------------------------------------------------------
# Record encoding
# ---------------------------------------------------------------------------

def _make_log_formatter() -> logging.Formatter:
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    return CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )


def _make_log_handler(log_file: str) -> logging.FileHandler:
    handler = logging.FileHandler(log_file)
    handler.setFormatter(_make_log_formatter())
    return handler


def init_logging(log_file: Optional[str] = None):
    """Custom logging format for better readability."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(_make_log_formatter())

    if log_file is not None:
        logger.addHandler(_make_log_handler(log_file))
        logging.info(f"Logging to file: {log_file}")


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
    """Load the policy and shard its parameters with FSDP, matching training."""
    logging.info(f"Loading policy '{config_name}' from '{checkpoint_path}' ...")
    config = _config.get_config(config_name)

    # Standard load — params are initially fully replicated across all devices.
    policy = policy_config.create_trained_policy(config, checkpoint_path)

    # PyTorch models are not JAX-sharded; nothing to do.
    if policy._is_pytorch_model:
        logging.info("PyTorch model detected; skipping JAX FSDP sharding.")
        return policy

    # Build the same FSDP mesh used during training so each device holds only
    # a slice of the parameters instead of a full replicated copy.
    mesh = _sharding.make_mesh(4)
    logging.info(
        f"FSDP mesh: shape={mesh.shape}, total devices={jax.device_count()}"
    )

    # Compute per-parameter FSDP sharding specs from the current (replicated) state.
    graphdef, state = nnx.split(policy._model)
    state_sharding = _sharding.fsdp_sharding(
        jax.eval_shape(lambda s: s, state), mesh, log=True
    )

    # Reshard parameters in-place and rebuild the model.
    sharded_state = jax.device_put(state, state_sharding)
    policy._model = nnx.merge(graphdef, sharded_state)

    # Rebuild the JIT so it captures the newly sharded state.
    policy._sample_actions = nnx_utils.module_jit(policy._model.sample_actions)

    logging.info("Policy loaded and sharded across FSDP mesh.")
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
    policy,
    *,
    resize: int,
    batch_size: int,
    max_episodes: Optional[int],
    max_steps_per_episode: Optional[int],
):
    """Iterate over one RLDS dataset, yielding (step_id, actions) per step.

    Observations are accumulated into groups of batch_size before a single
    batched model call; any remainder is flushed at the end of the dataset.
    The caller owns writing to disk and index management.
    """
    import dlimp as dl
    import tensorflow as tf
    import tensorflow_datasets as tfds

    # tf.config.set_visible_devices([], "TPU")

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

    first_traj = True
    logged_first_step = [False]  # mutable flag accessible from flush()

    # Pending batch: list of (step_id, obs) pairs not yet sent to the model.
    pending_ids: list[str] = []
    pending_obs: list[dict] = []

    def flush() -> list[tuple[str, np.ndarray]]:
        if not pending_obs:
            return []
        results = policy.infer_batch(pending_obs)
        out = []
        for step_id, obs, result in zip(pending_ids, pending_obs, results):
            actions = np.asarray(result["actions"], dtype=np.float32)
            if not logged_first_step[0]:
                logged_first_step[0] = True
                logging.info(f"  [step 0] obs keys: {list(obs.keys())}")
                for k, v in obs.items():
                    v_info = f"shape={v.shape} dtype={v.dtype}" if isinstance(v, np.ndarray) else repr(v)
                    logging.info(f"    obs[{k!r}]: {v_info}")
                logging.info(f"  [step 0] action chunk shape: {actions.shape}")
                logging.info(f"  [step 0] actions[:3]: {actions[:3]}")
                logging.info("=== End first trajectory debug ===")
            out.append((step_id, actions))
        pending_ids.clear()
        pending_obs.clear()
        return out

    for traj in tqdm.tqdm(
        dataset.as_numpy_iterator(),
        desc=f"[proc {proc_idx}/{num_procs}] {ds_name}:{ds_version}",
        unit="ep",
        miniters=10,
    ):
        # ---- Extract trajectory-level metadata ----
        ep_meta = traj["traj_metadata"]["episode_metadata"]
        recording_folder = ep_meta["recording_folderpath"][0].decode()
        file_path = ep_meta["file_path"][0].decode()

        # language_instruction may be a scalar bytes or a 1-D numpy array of bytes.
        lang = traj["language_instruction"]
        prompt = (lang.flat[0] if isinstance(lang, np.ndarray) else lang).decode()

        # ---- Per-step arrays ----
        ext_imgs_1 = traj["observation"]["exterior_image_1_left"]   # (T,) encoded
        wrist_imgs = traj["observation"]["wrist_image_left"]         # (T,) encoded
        joint_pos = traj["observation"]["joint_position"]            # (T, 7)
        gripper_pos = traj["observation"]["gripper_position"]        # (T, 1)

        if first_traj:
            first_traj = False
            logging.info("=== First trajectory debug ===")
            logging.info(f"  recording_folder : {recording_folder}")
            logging.info(f"  file_path        : {file_path}")
            logging.info(f"  prompt           : {prompt!r}")
            logging.info(f"  language_instruction raw type/shape: "
                         f"{type(traj['language_instruction'])}, "
                         f"{getattr(traj['language_instruction'], 'shape', 'n/a')}")
            logging.info(f"  joint_position   shape: {joint_pos.shape}, dtype: {joint_pos.dtype}")
            logging.info(f"  gripper_position shape: {gripper_pos.shape}, dtype: {gripper_pos.dtype}")
            logging.info(f"  ext_imgs_1       shape: {ext_imgs_1.shape}, dtype: {ext_imgs_1.dtype}")
            logging.info(f"  wrist_imgs       shape: {wrist_imgs.shape}, dtype: {wrist_imgs.dtype}")

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

            step_id = f"{recording_folder}--{file_path}--{t}"
            pending_ids.append(step_id)
            pending_obs.append(obs)

            if len(pending_obs) >= batch_size:
                yield from flush()

    # Flush any remaining steps that did not fill a full batch.
    yield from flush()


# ---------------------------------------------------------------------------
# Merge shards
# ---------------------------------------------------------------------------

def _download_from_gcs(gcs_path: str, local_path: pathlib.Path) -> None:
    """Download a single GCS file to a local path in 64 MB chunks."""
    import tensorflow as tf

    logging.info(f"Downloading {gcs_path} -> {local_path} ...")
    with tf.io.gfile.GFile(gcs_path, "rb") as f_in, open(local_path, "wb") as f_out:
        while chunk := f_in.read(64 * 1024 * 1024):
            f_out.write(chunk)


def merge_all_shards(
    local_dir: pathlib.Path,
    num_procs: int,
    gcs_dir: Optional[str] = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Merge per-process checkpoint parts into one ArrayRecord + one index JSON.

    Each process writes files named:
        proc{i:05d}_ckpt{j:05d}.arrayrecord   – one or more checkpoint parts
        proc{i:05d}_index.json                 – step_id → local_row (across all parts)

    If gcs_dir is set, those files are read from GCS; each checkpoint part is
    downloaded to a temporary local file, processed, then deleted immediately so
    that local disk usage is bounded to one part at a time (plus the growing
    merged output file).

    Returns (arrayrecord_path, index_json_path), both written to local_dir.
    """
    import tensorflow as tf

    merged_ar_path = local_dir / "relabeled_actions.arrayrecord"
    merged_idx_path = local_dir / "relabeled_actions_index.json"
    logging.info(f"Merging {num_procs} process shard(s) into {merged_ar_path} ...")

    merged_index: dict[str, int] = {}
    global_row = 0

    writer = ArrayRecordWriter(str(merged_ar_path), ARRAYRECORD_OPTIONS)
    try:
        for proc in tqdm.tqdm(range(num_procs), desc="Merging processes", unit="proc"):
            # ---- Load per-process index ----
            idx_name = f"proc{proc:05d}_index.json"
            if gcs_dir:
                with tf.io.gfile.GFile(tf.io.gfile.join(gcs_dir, idx_name)) as f:
                    proc_index: dict[str, int] = json.load(f)
            else:
                with open(local_dir / idx_name) as f:
                    proc_index: dict[str, int] = json.load(f)

            # Reverse: local_row → step_id
            row_to_step = {v: k for k, v in proc_index.items()}

            # ---- Discover checkpoint parts for this process (ordered) ----
            ckpt_glob = f"proc{proc:05d}_ckpt*.arrayrecord"
            if gcs_dir:
                ckpt_srcs = sorted(tf.io.gfile.glob(tf.io.gfile.join(gcs_dir, ckpt_glob)))
            else:
                ckpt_srcs = sorted(str(p) for p in local_dir.glob(ckpt_glob))

            if not ckpt_srcs:
                raise FileNotFoundError(f"No checkpoint parts found for process {proc}")

            # ---- Stream through parts one at a time ----
            local_row = 0
            for ckpt_src in tqdm.tqdm(ckpt_srcs, desc=f"  proc {proc}", unit="part", leave=False):
                tmp: Optional[pathlib.Path] = None
                if gcs_dir:
                    tmp = local_dir / pathlib.Path(ckpt_src).name
                    _download_from_gcs(ckpt_src, tmp)
                    ar_src = str(tmp)
                else:
                    ar_src = ckpt_src

                try:
                    reader = ArrayRecordReader(ar_src)
                    num_records = reader.num_records()
                    for start in range(0, num_records, MERGE_BATCH):
                        end = min(start + MERGE_BATCH, num_records)
                        for record in reader.read(list(range(start, end))):
                            writer.write(record)
                            merged_index[row_to_step[local_row]] = global_row
                            local_row += 1
                            global_row += 1
                    reader.close()
                finally:
                    if tmp is not None:
                        tmp.unlink(missing_ok=True)
    finally:
        writer.close()

    with open(merged_idx_path, "w") as f:
        json.dump(merged_index, f)

    logging.info(f"Merge complete: {global_row:,} records -> {merged_ar_path}")
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

    batch_size: int = 32
    """Number of steps to accumulate before a single batched model call."""

    checkpoint_steps: int = 50_000
    """Roll over to a new local shard file every this many steps, uploading and
    deleting the completed part when output_dir is a gs:// path.  This bounds
    local disk usage to roughly checkpoint_steps * bytes_per_record at a time.
    Set to 0 to write a single file and upload only at the end."""

    # Merge mode
    merge: bool = False
    """If True, merge existing per-process shards in output_dir and exit.
    Useful for re-running the merge step without re-running inference."""

    # Debugging limits
    max_episodes: Optional[int] = None
    max_steps_per_episode: Optional[int] = None


def main(args: Args) -> None:
    # Initialize JAX distributed if running in a multi-process context.
    # On a single TPU VM all devices are in one process; this is a no-op.
    if os.environ.get("JAX_COORDINATOR_ADDRESS") or os.environ.get("SLURM_NTASKS"):
        jax.distributed.initialize()

    proc_idx = jax.process_index()
    num_procs = jax.process_count()

    # ArrayRecord cannot stream writes to GCS; write shards locally and upload.
    # The temp dir is created in the current working directory so it is easy to
    # locate and is not silently cleaned up by the OS.
    is_gcs_output = args.output_dir.startswith("gs://")
    if is_gcs_output:
        local_dir = pathlib.Path(tempfile.mkdtemp(prefix="droid_relabel_", dir="."))
    else:
        local_dir = pathlib.Path(args.output_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Primary log inside the shard directory; secondary log in CWD for easy access.
    log_file = str(local_dir / f"relabel_proc{proc_idx:05d}.log")
    cwd_log_file = str(pathlib.Path(".") / f"relabel_proc{proc_idx:05d}.log")
    print(f"Initializing logging... (log files: {log_file}, {cwd_log_file})")
    init_logging(log_file=log_file)
    # Add the CWD log file as a second handler.
    logging.getLogger().addHandler(
        _make_log_handler(cwd_log_file)
    )
    logging.info(
        f"JAX: process {proc_idx}/{num_procs}, "
        f"local devices: {jax.local_device_count()}, "
        f"total devices: {jax.device_count()}"
    )

    if is_gcs_output:
        logging.info(f"GCS output detected; writing shards locally to {local_dir}")

    # ---- Merge-only mode ----
    if args.merge:
        mhu.sync_global_devices("pre_merge")
        if proc_idx == 0:
            merged_ar, merged_idx = merge_all_shards(
                local_dir, num_procs,
                gcs_dir=args.output_dir if is_gcs_output else None,
            )
            if is_gcs_output:
                _upload_to_gcs(merged_ar, args.output_dir)
                _upload_to_gcs(merged_idx, args.output_dir)
        mhu.sync_global_devices("post_merge")
        return

    # ---- Relabeling mode ----
    policy = load_policy(args.policy_config_name, args.checkpoint_path)

    index: dict[str, int] = {}
    total_steps = 0
    part_idx = 0

    def _ckpt_path(part: int) -> pathlib.Path:
        return local_dir / f"proc{proc_idx:05d}_ckpt{part:05d}.arrayrecord"

    ar_path = _ckpt_path(part_idx)
    writer = ArrayRecordWriter(str(ar_path), ARRAYRECORD_OPTIONS)
    steps_in_part = 0
    logging.info(f"[proc {proc_idx}] Writing checkpoint part {part_idx}: {ar_path}")

    def _roll_over() -> None:
        """Close current part, upload+delete if GCS, open the next part."""
        nonlocal writer, ar_path, part_idx, steps_in_part
        writer.close()
        if is_gcs_output:
            _upload_to_gcs(ar_path, args.output_dir)
            ar_path.unlink()
            logging.info(f"[proc {proc_idx}] Uploaded and deleted checkpoint part {part_idx}")
        part_idx += 1
        ar_path = _ckpt_path(part_idx)
        writer = ArrayRecordWriter(str(ar_path), ARRAYRECORD_OPTIONS)
        steps_in_part = 0
        logging.info(f"[proc {proc_idx}] Opened checkpoint part {part_idx}: {ar_path}")

    try:
        for ds_spec in args.datasets:
            if ":" not in ds_spec:
                raise ValueError(f"Dataset spec must be '<name>:<version>', got: {ds_spec!r}")
            ds_name, ds_version = ds_spec.split(":", 1)
            steps_before = total_steps
            for step_id, actions in relabel_dataset(
                ds_name=ds_name,
                ds_version=ds_version,
                data_dir=args.data_dir,
                policy=policy,
                resize=args.resize,
                batch_size=args.batch_size,
                max_episodes=args.max_episodes,
                max_steps_per_episode=args.max_steps_per_episode,
            ):
                index[step_id] = total_steps
                writer.write(encode_actions(actions))
                total_steps += 1
                steps_in_part += 1

                if args.checkpoint_steps > 0 and total_steps % args.checkpoint_steps == 0:
                    _roll_over()

            logging.info(
                f"[proc {proc_idx}] {ds_name}:{ds_version} -> {total_steps - steps_before:,} steps written"
            )
    finally:
        # Close and upload the last (possibly partial) checkpoint part.
        writer.close()
        if is_gcs_output and steps_in_part > 0:
            _upload_to_gcs(ar_path, args.output_dir)
            ar_path.unlink()
            logging.info(f"[proc {proc_idx}] Uploaded and deleted final checkpoint part {part_idx}")
        elif steps_in_part == 0 and ar_path.exists():
            ar_path.unlink()  # empty trailing file from exact rollover boundary

    logging.info(f"[proc {proc_idx}] Done. Total steps: {total_steps:,}")

    # Write and upload per-process index.
    idx_path = local_dir / f"proc{proc_idx:05d}_index.json"
    with open(idx_path, "w") as f:
        json.dump(index, f)
    if is_gcs_output:
        _upload_to_gcs(idx_path, args.output_dir)
        idx_path.unlink()

    # Wait for all processes to finish writing their shards.
    mhu.sync_global_devices("relabeling_complete")

    # Process 0 merges and uploads the final combined files.
    if proc_idx == 0:
        merged_ar, merged_idx = merge_all_shards(
            local_dir, num_procs,
            gcs_dir=args.output_dir if is_gcs_output else None,
        )
        if is_gcs_output:
            _upload_to_gcs(merged_ar, args.output_dir)
            _upload_to_gcs(merged_idx, args.output_dir)

    mhu.sync_global_devices("merge_complete")
    logging.info(f"[proc {proc_idx}] All done.")


if __name__ == "__main__":
    tyro.cli(main)

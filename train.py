import functools
from loguru import logger
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast, Dict
from bsr_dataset import get_datasets,preprocessing_pipeline
import jax
import jax.experimental.compilation_cache.compilation_cache
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from chex import PRNGKey
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import random
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.stages import Compiled, Wrapped
from bs_roformer import BSRoformer
from profiling import memory_usage_params
import audax.core.stft
from einops import einsum, rearrange, pack, unpack,repeat
from omegaconf import OmegaConf
import argparse

jax.experimental.compilation_cache.compilation_cache.set_cache_dir("jit_cache")


    
class Trainer:

    def __init__(
        self,
        rng: PRNGKey,
        hp : Any
    ) -> None:
        self.init_step = 0
        self.optimizer = optax.chain(
            optax.adamw(learning_rate=hp.train.learning_rate,b1=hp.train.betas[0],b2=hp.train.betas[1]),
        )
        init_key, self.train_key = random.split(rng, 2)

        #dtype = jnp.float16 if half_precision else jnp.float32
        
        self.bsr_model = BSRoformer(dim=hp.model.dim,
                                    depth=hp.model.depth,
                                    stereo=hp.model.stereo,
                                    time_transformer_depth=hp.model.time_transformer_depth,
                                    freq_transformer_depth=hp.model.freq_transformer_depth)
        
        n_devices = len(jax.devices())

        # x, y, t
        bsr_input_values = (
            jnp.ones((n_devices,2, 44100)),
        )
        def create_bsr_train_state(x,model, optimizer):
            variables = model.init(
                {"params": init_key, "dropout": init_key},
                raw_audio=x,
                deterministic=True
            )
            train_state = TrainState.create(
                apply_fn=self.bsr_model.apply,
                params=variables["params"],
                tx=optimizer
            )
            return train_state

        if jax.process_index() == 0:
            logger.info(f"Available devices: {jax.devices()}")

        # Create a device mesh according to the physical layout of the devices.
        # device_mesh is just an ndarray
        device_mesh = mesh_utils.create_device_mesh((n_devices, 1))

        if jax.process_index() == 0:
            logger.info(f"Device mesh: {device_mesh}")

        # Async checkpointer for saving checkpoints across processes
        #base_dir_abs = os.getcwd()
        options = ocp.CheckpointManagerOptions(max_to_keep=3)
        self.checkpoint_manager = ocp.CheckpointManager(
            hp.log.pth_dir,
            #f"{base_dir_abs}/checkpoints", 
            options=options
        )

        # The axes are (data, model), so the mesh is (n_devices, 1) as the model is replicated across devices.
        # This object corresponds the axis names to the layout of the physical devices,
        # so that sharding a tensor along the axes shards according to the corresponding device_mesh layout.
        # i.e. with device layout of (8, 1), data would be replicated to all devices, and model would be replicated to 1 device.
        self.mesh = Mesh(device_mesh, axis_names=("data", "model"))
        if jax.process_index() == 0:
            logger.info(f"Mesh: {self.mesh}")



        def get_sharding_for_spec(pspec: PartitionSpec) -> NamedSharding:
            """
            Get a NamedSharding for a given PartitionSpec, and the device mesh.
            A NamedSharding is simply a combination of a PartitionSpec and a Mesh instance.
            """
            return NamedSharding(self.mesh, pspec)

        # This shards the first dimension of the input data (batch dim) across the data axis of the mesh.
        x_sharding = get_sharding_for_spec(PartitionSpec("data"))

        # Returns a pytree of shapes for the train state
        bsr_train_state_sharding_shape = jax.eval_shape(
            functools.partial(
                create_bsr_train_state, model=self.bsr_model, optimizer=self.optimizer
            ),
            *bsr_input_values,
        )

        # Get the PartitionSpec for all the variables in the train state
        bsr_train_state_sharding = nn.get_sharding(bsr_train_state_sharding_shape, self.mesh)
        bsr_input_sharding: Any = (x_sharding,)
        if jax.process_index() == 0:
            logger.info(f"Initializing model...")
        # Shard the train_state so so that it's replicated across devices
        jit_create_bsr_train_state_fn = jax.jit(
            create_bsr_train_state,
            static_argnums=(1,2),
            in_shardings=bsr_input_sharding,  # type: ignore
            out_shardings=bsr_train_state_sharding,
        )
        self.bsr_train_state = jit_create_bsr_train_state_fn(
            *bsr_input_values, self.bsr_model, self.optimizer
        )
        bsr_total_bytes, bsr_total_params = memory_usage_params(self.bsr_train_state.params)
        if jax.process_index() == 0:
            logger.info(f"BSR Model parameter count: {bsr_total_params} using: {bsr_total_bytes}")

            logger.info("JIT compiling step functions...")

        bsr_step_in_sharding: Any = (
            bsr_train_state_sharding,
            x_sharding,
            x_sharding,
            None
        )
        bsr_step_out_sharding: Any = (
            get_sharding_for_spec(PartitionSpec()),
            bsr_train_state_sharding,
        )

        def extract_step(
            state: TrainState,
            raw_audio: jnp.ndarray,
            target_audio : jnp.ndarray,
            prng_key : jnp.ndarray,
            deterministic : bool
        ) -> Tuple[jnp.ndarray, TrainState]:
            prng_key, step_key,dropout_key = random.split(prng_key,3)
            predict_audio = state.apply_fn({'params': state.params},
                                           raw_audio,
                                           deterministic=deterministic,
                                           rngs={'rnorms':step_key,'dropout': dropout_key})
            predict_audio = predict_audio[...,:target_audio.shape[-1]]

            loss = jnp.mean(jnp.abs(predict_audio - target_audio))

            multi_stft_resolution_loss = 0.

            for window_size in hp.model.multi_stft_resolutions_window_sizes:
                res_stft_kwargs = dict(
                    n_fft=max(window_size, hp.model.stft_n_fft),  # not sure what n_fft is across multi resolution stft
                    win_length=window_size,
                    window=jnp.hanning(window_size),
                    hop_length=hp.model.multi_stft_hop_size
                )

                recon_Y = audax.core.stft.stft(rearrange(predict_audio, '... s t -> (... s) t'), **res_stft_kwargs)
                target_Y = audax.core.stft.stft(rearrange(target_audio, '... s t -> (... s) t'), **res_stft_kwargs)

                multi_stft_resolution_loss = multi_stft_resolution_loss + jnp.mean(jnp.abs(recon_Y - target_Y))


            total_loss = loss + multi_stft_resolution_loss
            return total_loss, state
        
        self.bsr_train_step: Wrapped = jax.jit(
            functools.partial(extract_step,deterministic=False),
            in_shardings=bsr_step_in_sharding,
            out_shardings=bsr_step_out_sharding,
        )
        # self.bsr_eval_step: Wrapped = jax.jit(
        #     functools.partial(bsr_train_step, training=False),
        #     in_shardings=bsr_step_in_sharding,
        #     out_shardings=bsr_step_out_sharding,
        # )

        # if profile:
        #     if jax.process_index() == 0:
        #         logger.info("AOT compiling step functions...")
        #     compiled_step: Compiled = self.diff_train_step.lower(
        #         self.diff_train_state, *diff_input_values[:2], init_key
        #     ).compile()
        #     train_cost_analysis: Dict = compiled_step.cost_analysis()[0]  # type: ignore
        #     self.flops_for_step = train_cost_analysis.get("flops", 0)
        #     if jax.process_index() == 0:
        #         logger.info(
        #             f"Steps compiled, train cost analysis FLOPs: {self.flops_for_step}"
        #         )
        # else:
        self.flops_for_step = 0

    def save_checkpoint(self, global_step: int):
        if self.bsr_train_state is not None:
            self.checkpoint_manager.save(global_step,args=ocp.args.StandardSave(self.bsr_train_state))
    def restore_checkpoint(self):
        step = self.checkpoint_manager.latest_step() 
        self.bsr_train_state=self.checkpoint_manager.restore(step)
        self.init_step = step + 1

def main(args):
    """
    Arguments:
        n_epochs: Number of epochs to train for.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        eval_save_steps: Number of steps between evaluation runs and checkpoint saves.
        n_eval_batches: Number of batches to evaluate on.
        sample_every_n: Number of epochs between sampling runs.
        dataset_name: Name of the dataset config to select, valid options are in DATASET_CONFIGS.
        profile: Run a single train and eval step, and print out the cost analysis, then exit.
        half_precision: case the model to fp16 for training.
    """
    if args.hardware == "tpu":
        jax.distributed.initialize()
    hp = OmegaConf.load(args.config)
    rng = random.PRNGKey(hp.train.seed)
    trainer = Trainer(rng, hp)

    if trainer.checkpoint_manager.latest_step() is not None:
        trainer.restore_checkpoint()

    dataset = get_datasets(hp.data_loader.dataset_path)
    data_iterator = preprocessing_pipeline(dataset=dataset,
                                           global_batch_size=hp.data_loader.global_batch_size,
                                           global_mesh=trainer.mesh,
                                           segment_length=hp.data.segment_size,
                                           grain_worker_count=hp.data_loader.worker_count,
                                           dataloading_host_index=jax.process_index(),
                                           dataloading_host_count=hp.data_loader.host_number,
                                           data_columns=hp.data.data_columns,
                                           shuffle=hp.data_loader.shuffle,
                                           data_shuffle_seed=hp.train.seed,
                                           num_epochs=hp.data_loader.num_epochs,
                                           drop_remainder=hp.data_loader.drop_remainder)
    example_batch = None

    for step in range(trainer.init_step, hp.train.total_steps):
        example_batch = next(data_iterator)

        # Train step
        step_key = jax.jit(jax.random.fold_in)(rng, step)

        bsr_train_loss, bsr_updated_state = trainer.bsr_train_step(
            trainer.bsr_train_state, 
            example_batch['mixture'], 
            example_batch['vocals'],
            step_key
        )
        trainer.bsr_train_state = bsr_updated_state
            
        if step % hp.log.info_interval == 0:
            if jax.process_index() == 0:
                logger.info(f"step: {step} bsr_train_loss: {bsr_train_loss}")


        #     summary_writer.add_scalar(
        #         "train_step_time",
        #         step_duration,
        #         global_step,
        #     )

        # summary_writer.add_scalar("train_loss", train_loss, global_step)

        if step % hp.log.eval_interval == 0:
            trainer.save_checkpoint(step)

        # if profile:
        #     logger.info("\nExiting after profiling a single step.")
        #     return

if __name__ == "__main__":
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/base.yaml', help="Config File for Train")
    parser.add_argument("--hardware",default="tpu", type=str, help="Hardware Type")
    args = parser.parse_args()
    main(args)
    
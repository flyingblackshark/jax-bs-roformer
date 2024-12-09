import argparse
import flax
from bs_roformer import BSRoformer
from convert import load_params
import librosa
import numpy as np
import jax.numpy as jnp
import soundfile as sf
import glob
import os
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils
from functools import partial
import jax
from jax.experimental.compilation_cache import compilation_cache as cc
import time
from omegaconf import OmegaConf
cc.set_cache_dir("./jax_cache")
def run_folder(args):
    hp = OmegaConf.load(args.config_path)
    start_time = time.time()
    model = BSRoformer(dim=hp.model.dim,
                        depth=hp.model.depth,
                        stereo=hp.model.stereo,
                        time_transformer_depth=hp.model.time_transformer_depth,
                        freq_transformer_depth=hp.model.freq_transformer_depth)
    params = load_params(args.start_check_point,hp)
    model = (model,params)
    
    all_mixtures_path = glob.glob(args.input_folder + '/*.*')
    all_mixtures_path.sort()
    print('Total files found: {}'.format(len(all_mixtures_path)))

    # instruments = config.training.instruments
    # if config.training.target_instrument is not None:
    #     instruments = [config.training.target_instrument]

    if not os.path.isdir(args.store_dir):
        os.mkdir(args.store_dir)

    # if not verbose:
    #     all_mixtures_path = tqdm(all_mixtures_path, desc="Total progress")
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices=device_mesh, axis_names=('data'))
    for path in all_mixtures_path:
        print("Starting processing track: ", path)
        try:
            mix, sr = librosa.load(path, sr=44100, mono=False)
        except Exception as e:
            print('Can read track: {}'.format(path))
            print('Error message: {}'.format(str(e)))
            continue

        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=0)

        #mix_orig = mix.copy()

        res = demix_track(model,mix,mesh,hp)
        
        estimates = res.squeeze(0)
        #estimates = estimates/1024.
        estimates = estimates.transpose(1,0)
        
        file_name, _ = os.path.splitext(os.path.basename(path))
        output_file = os.path.join(args.store_dir, f"{file_name}_dereverb.wav")
        sf.write(output_file, estimates, sr, subtype = 'FLOAT')

        # file_name, _ = os.path.splitext(os.path.basename(path))
        # instrum_file_name = os.path.join(args.store_dir, f"{file_name}_instrumental.wav")
        # sf.write(instrum_file_name, mix_orig.T - estimates, sr, subtype = 'FLOAT')

    #time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))


def demix_track(model, mix,mesh, hp):
    model , params = model
    #default chunk size 
    C = hp.inference.chunk_size
    N = hp.inference.num_overlap
    fade_size = C // 10
    step = int(C // N)
    border = C - step
    batch_size = hp.inference.batch_size

    x_sharding = NamedSharding(mesh,PartitionSpec('data'))
    @partial(jax.jit, in_shardings=(None,x_sharding),
                    out_shardings=x_sharding)
    def model_apply(params, x):
        return model.apply({'params': params}, x , deterministic=True)
    length_init = mix.shape[-1]

    # Do pad from the beginning and end to account floating window results better
    if length_init > 2 * border and (border > 0):
        mix = np.pad(mix, ((0,0),(border, border)), mode='reflect')
    def _getWindowingArray(window_size, fade_size):
        fadein = np.linspace(0, 1, fade_size)
        fadeout = np.linspace(1, 0, fade_size)
        window = np.ones(window_size)
        window[-fade_size:] = (window[-fade_size:]*fadeout)
        window[:fade_size] = (window[:fade_size]*fadein)
        return window
    # windowingArray crossfades at segment boundaries to mitigate clicking artifacts
    windowingArray = _getWindowingArray(C, fade_size)

   
    # if config.training.target_instrument is not None:
    req_shape = (1, ) + tuple(mix.shape)
    # else:
    #     req_shape = (len(config.training.instruments),) + tuple(mix.shape)

    result = np.zeros(req_shape, dtype=jnp.float32)
    counter = np.zeros(req_shape, dtype=jnp.float32)
    i = 0
    batch_data = []
    batch_locations = []

    while i < mix.shape[1]:
        part = mix[:, i:i + C]
        length = part.shape[-1]
        if length < C:
            if length > C // 2 + 1:
                part = np.pad(part,((0,0),(0,C-length)),mode='reflect')
            else:
                part = np.pad(part,((0,0),(0,C-length)),mode='constant')
        batch_data.append(part)
        batch_locations.append((i, length))
        i += step

        if len(batch_data) >= batch_size or (i >= mix.shape[1]):
            arr = np.stack(batch_data, axis=0)
            B_padding = max((batch_size-len(batch_data)),0)
            arr = np.pad(arr,((0,B_padding),(0,0),(0,0)))

            # infer
            with mesh:
                arr = jnp.asarray(arr)
                x = model_apply(params,arr)
            window = windowingArray
            if i - step == 0:  # First audio chunk, no fadein
                window[:fade_size] =1
            elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                window[-fade_size:] =1
            
            total_add_value = jax.jit(jnp.multiply, in_shardings=(x_sharding,None),out_shardings=x_sharding)(x[..., :C],window)
            total_add_value = total_add_value[:batch_size-B_padding]
            total_add_value = np.asarray(total_add_value)
            for j in range(len(batch_locations)):
                start, l = batch_locations[j]
                result[..., start:start+l] += total_add_value[j][..., :l]
                counter[..., start:start+l]+= window[..., :l]

            batch_data = []
            batch_locations = []

    estimated_sources = result / counter
    np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    if length_init > 2 * border and (border > 0):
        # Remove pad
        estimated_sources = estimated_sources[..., border:-border]
    return estimated_sources

def proc_folder(args):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_type", type=str, default='mdx23c', 
    #                     help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer, scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    parser.add_argument("--config_path", type=str, default='./configs/base.yaml', help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='deverb_bs_roformer_8_256dim_8depth.ckpt', help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder",default="./input", type=str, help="folder with mixtures to process")
    parser.add_argument("--store_dir", default="./output", type=str, help="path to store results as wav file")
    # parser.add_argument("--disable_detailed_pbar", action='store_true', help="disable detailed progress bar")
    args = parser.parse_args()
    run_folder(args)


if __name__ == "__main__":
    proc_folder(None)


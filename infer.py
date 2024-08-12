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
import jaxloudnorm as jln
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils
from functools import partial
import jax
from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("./jax_cache")
def run_folder(args,verbose=False):
    model = BSRoformer(256,8,precision=jax.lax.Precision.DEFAULT)
    params = load_params()
    model = (model,params)
    
    all_mixtures_path = glob.glob(args.input_folder + '/*.*')
    all_mixtures_path.sort()
    print('Total files found: {}'.format(len(all_mixtures_path)))

    # instruments = config.training.instruments
    # if config.training.target_instrument is not None:
    #     instruments = [config.training.target_instrument]

    if not os.path.isdir(args.store_dir):
        os.mkdir(args.store_dir)

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path, desc="Total progress")

    # if args.disable_detailed_pbar:
    #     detailed_pbar = False
    # else:
    #     detailed_pbar = True
    device_mesh = mesh_utils.create_device_mesh((4,))
    mesh = Mesh(devices=device_mesh, axis_names=('data'))
    for path in all_mixtures_path:
        print("Starting processing track: ", path)
        if not verbose:
            all_mixtures_path.set_postfix({'track': os.path.basename(path)})
        try:
            mix, sr = librosa.load(path, sr=44100, mono=False)
        except Exception as e:
            print('Can read track: {}'.format(path))
            print('Error message: {}'.format(str(e)))
            continue
        meter = jln.Meter(sr) # create BS.1770 meter
        loudness_old = meter.integrated_loudness(mix.transpose(1,0))
        if loudness_old > -16:
            loudness_old -= 2

        if len(mix.shape) == 1:
            mix = jnp.stack([mix, mix], axis=0)

        res = demix_track(model,mix,mesh, pbar=False)
        # for instr in instruments:
        
        estimates = res.squeeze(0)
        estimates = estimates.transpose(1,0)
        loudness_new = meter.integrated_loudness(estimates)
        estimates = jln.normalize.loudness(estimates, loudness_new, loudness_old)
        
        # if 'normalize' in config.inference:
        #     if config.inference['normalize'] is True:
        #       estimates = estimates * std + mean
        file_name, _ = os.path.splitext(os.path.basename(path))
        output_file = os.path.join(args.store_dir, f"{file_name}_dereverb.wav")
        sf.write(output_file, estimates, sr, subtype = 'FLOAT')

        # # Output "instrumental", which is an inverse of 'vocals'
        # if 'vocals' in instruments and args.extract_instrumental:
        #     file_name, _ = os.path.splitext(os.path.basename(path))
        #     instrum_file_name = os.path.join(args.store_dir, f"{file_name}_instrumental.wav")
        #     estimates = res['vocals'].T
        #     if 'normalize' in config.inference:
        #         if config.inference['normalize'] is True:
        #             estimates = estimates * std + mean
        #     sf.write(instrum_file_name, mix_orig.T - estimates, sr, subtype = 'FLOAT')

    #time.sleep(1)
    #print("Elapsed time: {:.2f} sec".format(time.time() - start_time))
def _getWindowingArray(window_size, fade_size):
    fadein = jnp.linspace(0, 1, fade_size)
    fadeout = jnp.linspace(1, 0, fade_size)
    window = jnp.ones(window_size)
    window = window.at[-fade_size:].set(window[-fade_size:]*fadeout)
    window = window.at[:fade_size].set(window[:fade_size]*fadein)
    return window

def demix_track(model, mix,mesh, pbar=False):
    model , params = model
    C = 352768 #config.audio.chunk_size
    N = 4 #config.inference.num_overlap
    fade_size = C // 10
    step = int(C // N)
    border = C - step
    batch_size = 4 #config.inference.batch_size

    x_sharding = NamedSharding(mesh,PartitionSpec('data'))
    @partial(jax.jit, in_shardings=(None,x_sharding),
                    out_shardings=x_sharding)
    def model_apply(params, x):
        return model.apply({'params': params}, x , deterministic=True)
    length_init = mix.shape[-1]

    # Do pad from the beginning and end to account floating window results better
    if length_init > 2 * border and (border > 0):
        mix =jnp.pad(mix, ((0,0),(border, border)), mode='reflect')

    # windowingArray crossfades at segment boundaries to mitigate clicking artifacts
    windowingArray = _getWindowingArray(C, fade_size)

   
    # if config.training.target_instrument is not None:
    req_shape = (1, ) + tuple(mix.shape)
    # else:
    #     req_shape = (len(config.training.instruments),) + tuple(mix.shape)

    result = jnp.zeros(req_shape, dtype=jnp.float32)
    counter = jnp.zeros(req_shape, dtype=jnp.float32)
    i = 0
    batch_data = []
    batch_locations = []
    progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False) if pbar else None

    while i < mix.shape[1]:
        # print(i, i + C, mix.shape[1])
        part = mix[:, i:i + C]
        length = part.shape[-1]
        if length < C:
            if length > C // 2 + 1:
                part = jnp.pad(part,((0,0),(0,C-length)),mode='reflect')
                #part = nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
            else:
                part = jnp.pad(part,((0,0),(0,C-length)),mode='constant')
                #part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
        batch_data.append(part)
        batch_locations.append((i, length))
        i += step

        if len(batch_data) >= batch_size or (i >= mix.shape[1]):
            arr = jnp.stack(batch_data, axis=0)
            #x = model.apply({"params":params},arr,deterministic=True)
            B_padding = max((batch_size-len(batch_data)),0)
            arr = jnp.pad(arr,((0,B_padding),(0,0),(0,0)))
            with mesh:
                x = model_apply(params,arr)
            #x = model(arr)
            x = x[:batch_size-B_padding]
            window = windowingArray
            if i - step == 0:  # First audio chunk, no fadein
                window = window.at[:fade_size].set(1)
            elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                window = window.at[-fade_size:].set(1)

            for j in range(len(batch_locations)):
                start, l = batch_locations[j]
                result = result.at[..., start:start+l].set(result[..., start:start+l] + x[j][..., :l] * window[..., :l])
                counter = counter.at[..., start:start+l].set(counter[..., start:start+l] + window[..., :l])

            batch_data = []
            batch_locations = []

        if progress_bar:
            progress_bar.update(step)

    if progress_bar:
        progress_bar.close()

    estimated_sources = result / counter
    estimated_sources = jnp.nan_to_num(estimated_sources,copy=False,nan=0)
    #np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    if length_init > 2 * border and (border > 0):
        # Remove pad
        estimated_sources = estimated_sources[..., border:-border]
    return estimated_sources
    # if config.training.target_instrument is None:
    #     return {k: v for k, v in zip(config.training.instruments, estimated_sources)}
    # else:
    #     return {k: v for k, v in zip([config.training.target_instrument], estimated_sources)}
def proc_folder(args):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_type", type=str, default='mdx23c', 
    #                     help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer, scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    #parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='deverb_bs_roformer_8_256dim_8depth.ckpt', help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder",default="./input", type=str, help="folder with mixtures to process")
    parser.add_argument("--store_dir", default="./output", type=str, help="path to store results as wav file")
    parser.add_argument("--disable_detailed_pbar", action='store_true', help="disable detailed progress bar")
    args = parser.parse_args()
    run_folder(args,verbose=True)


if __name__ == "__main__":
    proc_folder(None)


from bs_roformer import BSRoformer
import librosa
import numpy as np
import jax.numpy as jnp
import soundfile as sf
test = BSRoformer(256,8)
from convert import load_params
params = load_params()
mix, sr = librosa.load("test.wav", sr=44100, mono=False)
mix = jnp.asarray(mix)
mix = jnp.expand_dims(mix,0)
output = test.apply({"params":params},mix,deterministic=True)
output = output.squeeze(0)
output = np.asarray(output)
output = librosa.to_mono(output)
output = output/np.max(output)
sf.write("output.wav", output, sr, subtype = 'FLOAT')
print(output)
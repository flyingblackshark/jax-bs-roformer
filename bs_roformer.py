from typing import Any, Sequence, Tuple
import audax.core
import audax.core.stft
from einops import einsum, rearrange, pack, unpack,repeat
from flax import traverse_util
import jax
import jax.numpy as jnp
import jax.lax as lax
import flax.linen as nn
import audax
import numpy as np
import util
def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

class RotaryEmbedding(nn.Module):
  """RoPE

  Attributes:
    min_timescale: Start of the geometric index. Determines the periodicity of
      the added signal.
    max_timescale: End of the geometric index. Determines the frequency of the
      added signal.
    embedding_dims: Dimension of the embedding to be generated.
  """

  min_timescale: int
  max_timescale: int
  embedding_dims: int = 0
  cast_as_fprop_dtype: bool = True
  fprop_dtype: jnp.dtype = jnp.bfloat16

  def setup(self) -> None:
    if self.embedding_dims % 2:
      raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")

  def __call__(
      self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      inputs: jax.Array,
      position: jax.Array,
  ) -> jax.Array:
    """Generates a jax.Array of sinusoids with different frequencies.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, S, N, H].
      position: Optional position jax.Array which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [B, S].

    Returns:
      a jax.Array of shape [B, S, N, H] which includes the inputs together with
      the rotary position embedding incorporated in it.
    """
    assert position is not None
    if len(inputs.shape) != 4:
      raise ValueError("Input is assumed to be a rank 4 tensor of shape" "[batch, sequence, heads, dims].")
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
          "The embedding dims of the rotary position embedding" "must match the hidden dimension of the inputs."
      )
    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
    position = position[:, :, jnp.newaxis, jnp.newaxis]
    sinusoid_inp = position / timescale
    sin = jnp.sin(sinusoid_inp).astype(inputs.dtype)
    cos = jnp.cos(sinusoid_inp).astype(inputs.dtype)
    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    if self.cast_as_fprop_dtype:
      first_part = first_part.astype(self.fprop_dtype)
      second_part = second_part.astype(self.fprop_dtype)
    x_out = jnp.concatenate((first_part, second_part), axis=-1)
    return x_out
  
class RMSNorm(nn.Module):
  """RMS normalization."""
  dim : int
  #epsilon: float = 1e-12
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  #kernel_axes: Tuple[str, ...] = ()
  #scale_init: Initializer = nn.initializers.ones

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    #x = jnp.asarray(x, jnp.float32)
    #features = x.shape[-1]
    #mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    #y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    x = x / jnp.linalg.norm(x,axis=-1,keepdims=True)
    gamma = self.param(
        "gamma",
        nn.initializers.ones,
        (self.dim,),
        self.weight_dtype,
    )

    gamma = jnp.asarray(gamma, self.dtype)
    y = x * gamma * (self.dim ** 0.5)
    return y

class FeedForward(nn.Module):
    dim:int
    mult:int=4
    dropout:float=0.
    @nn.compact
    def __call__(self, x,deterministic):
        dim_inner = int(self.dim * self.mult)
        net = nn.Sequential([
            RMSNorm(self.dim),
            nn.Dense(dim_inner),
            nn.gelu,
            nn.Dropout(self.dropout,deterministic=deterministic),
            nn.Dense(self.dim),
            nn.Dropout(self.dropout,deterministic=deterministic)]
        )
        return net(x)
class Attend(nn.Module):
    dropout:float = 0.
    #precision:jax.lax.Precision=jax.lax.Precision.HIGHEST
    #scale = None
    def setup(self):
        self.attn_dropout = nn.Dropout(self.dropout)

    def __call__(self, q, k, v,deterministic):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        #q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        #scale = default(self.scale, q.shape[-1] ** -0.5)
        scale = q.shape[-1] ** -0.5
        q = q.transpose(0,2,1,3)
        k = k.transpose(0,2,1,3)
        v = v.transpose(0,2,1,3)
        out = jax.nn.dot_product_attention(q,k,v,scale=scale)

        return out.transpose(0,2,1,3)
        # similarity

        # sim = einsum(q, k,f"b h i d, b h j d -> b h i j") * scale

        # # attention

        # attn = nn.softmax(sim,axis=-1)
        # attn = self.attn_dropout(attn,deterministic=deterministic)

        # # aggregate values

        # out = einsum(attn, v,f"b h i j, b h j d -> b h i d")

        # return out

def get_seq_pos(seq_len, offset = 0):

    return (jnp.arange(seq_len) + offset)# / self.interpolate_factor
def embed_forward(
        t : jnp.ndarray,
        seq_len = None,
        offset = 0,
        dim_head = None,
        freqs = None
    ):
        dim = dim_head
        theta = 10000
        #freqs = freqs = 1. / (theta ** (jnp.arange(0, dim, 2)[:(dim // 2)] / dim))

        freqs = einsum(t, freqs,'..., f -> ... f')
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        return freqs
def rotate_queries_or_keys(t, seq_dim = None, offset = 0, scale = None ,dim_head=None,freqs=None):
    #seq_dim = default(seq_dim, self.default_seq_dim)
    seq_dim = -2
    #assert not self.use_xpos or exists(scale), 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'
    seq_len = t.shape[seq_dim]

    seq = get_seq_pos(seq_len, offset = offset)

    freqs = embed_forward(seq, seq_len = seq_len, offset = offset , dim_head=dim_head,freqs=freqs)

    if seq_dim == -3:
        freqs = rearrange(freqs, 'n d -> n 1 d')

    return apply_rotary_emb(freqs, t, scale = default(scale, 1.), seq_dim = seq_dim)
def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = jax_unstack(x,axis= -1)
    x = jnp.stack((-x2, x1), axis = -1)
    return rearrange(x, '... d r -> ... (d r)')
def apply_rotary_emb(freqs, t, start_index = 0, scale = 1., seq_dim = -2):

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    #assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * jnp.cos(freqs) * scale) + (rotate_half(t) * jnp.sin(freqs) * scale)
    out = jnp.concatenate((t_left, t, t_right), axis = -1)

    return out
class Attention(nn.Module):
    dim:int
    heads:int=8
    dim_head:int=64
    dropout:float=0.
    #precision:jax.lax.Precision=jax.lax.Precision.HIGHEST
    @nn.compact
    def __call__(self, x,deterministic):
        dim_inner = self.heads * self.dim_head
        x = RMSNorm(self.dim)(x)
        temp_x = nn.Dense(dim_inner * 3, use_bias=False,name="to_qkv")(x)
        q, k, v = rearrange(temp_x, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        freqs = self.param(
        "freqs",
        nn.initializers.ones,
        (self.dim_head//2,),
        jnp.float32,
        )

        #if exists(self.rotary_embed):
        q = rotate_queries_or_keys(q,dim_head=self.dim_head,freqs=freqs)
        k = rotate_queries_or_keys(k,dim_head=self.dim_head,freqs=freqs)

        out = Attend(dropout=self.dropout,name="attend")(q, k, v,deterministic)

        gates = nn.Dense(self.heads,name="to_gates")(x)
        out = out * nn.sigmoid(rearrange(gates, 'b n h -> b h n 1'))

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = nn.Dense(self.dim, use_bias=False,name="to_out")(out)
        out = nn.Dropout(self.dropout)(out,deterministic=deterministic)
        return out
    
class Transformer(nn.Module):
    dim:int
    depth:int
    dim_head:int=64
    heads:int=8
    attn_dropout:float=0.
    ff_dropout:float=0.
    ff_mult:int=4
    norm_output:bool=True
    def setup(self):
        layers = []

        for _ in range(self.depth):
            attn = Attention(dim=self.dim, dim_head=self.dim_head, heads=self.heads, dropout=self.attn_dropout)

            layers.append([
                attn,
                FeedForward(dim=self.dim, mult=self.ff_mult, dropout=self.ff_dropout)
            ])

        self.layers = layers

    def __call__(self, x,deterministic):

        for attn, ff in self.layers:
            x = attn(x,deterministic) + x
            x = ff(x,deterministic) + x

        return x
class BandSplit(nn.Module):
    dim:int
    freqs_per_bands_with_complex: Sequence[int]
      
    @nn.compact
    def __call__(self, x):
        to_features = []
        freqs_per_bands_with_complex_cum = np.cumsum(self.freqs_per_bands_with_complex)
        for dim_in in self.freqs_per_bands_with_complex:
          net = nn.Sequential([
              RMSNorm(dim_in),
              nn.Dense(self.dim)
          ])

          to_features.append(net)
        
        x = jnp.split(x,freqs_per_bands_with_complex_cum, axis=-1)

        outs = []
        for split_input, to_feature in zip(x, to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return jnp.stack(outs, axis=-2)
def MLP(
    dim_in,
    dim_out,
    dim_hidden=None,
    depth=1,
    activation=nn.tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Dense(layer_dim_out))

        if is_last:
            continue

        net.append(activation)

    return nn.Sequential(net)
def jax_unstack(x, axis=0):
  return [lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]
class MaskEstimator(nn.Module):
    dim:int
    dim_inputs: Sequence[int]
    depth:int
    mlp_expansion_factor:int=4
    def setup(self):
        to_freqs = []
        dim_hidden = self.dim * self.mlp_expansion_factor

        for dim_in in self.dim_inputs:
            #net = []
            mlp_layer = MLP(self.dim, dim_in * 2, dim_hidden=dim_hidden, depth=self.depth)
            mlp = nn.Sequential([
                mlp_layer,
                nn.glu]
            )

            to_freqs.append(mlp)
        self.to_freqs = to_freqs

    def __call__(self, x):
        x = jax_unstack(x,axis=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return jnp.concatenate(outs, axis=-1)
DEFAULT_FREQS_PER_BANDS = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
]


class BSRoformer(nn.Module):
    dim:int=256
    depth:int=8
    stereo:bool=True
    num_stems:int=1
    time_transformer_depth:int=1
    freq_transformer_depth:int=1
    linear_transformer_depth:int=0
    #freqs_per_bands: Sequence[int] = DEFAULT_FREQS_PER_BANDS,
    # in the paper, they divide into ~60 bands, test with 1 for starters
    dim_head:int=64
    heads:int=8
    attn_dropout:float=0.1
    ff_dropout:float=0.1
    #flash_attn=True
    dim_freqs_in:int=1025
    stft_n_fft:int=2048
    stft_hop_length:int=512
    # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
    stft_win_length:int=2048
    stft_normalized:int=False
    #stft_window_fn: Optional[Callable] = None,
    mask_estimator_depth:int=2
    multi_stft_resolution_loss_weight:float=1.
    multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256)
    multi_stft_hop_size:int=147
    multi_stft_normalized:bool=False
    #precision : jax.lax.Precision = jax.lax.Precision.HIGHEST
    # multi_stft_window_fn: Callable = torch.hann_window
    @nn.compact
    def __call__(
            self,
            raw_audio,
            #target=None,
            #return_loss_breakdown=False,
            deterministic=False):
        audio_channels = 2 if self.stereo else 1

        if raw_audio.ndim == 2:
          raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (
                    self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # to stft

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

        stft_window = jnp.hanning(self.stft_win_length)

        stft_repr = audax.core.stft.stft(raw_audio, n_fft=self.stft_n_fft,hop_length=self.stft_hop_length,win_length=self.stft_win_length, window=stft_window)
        stft_repr = stft_repr.transpose(0,2,1)
        stft_repr = as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')
        stft_repr = rearrange(stft_repr,'b s f t c -> b (f s) t c')  # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting

        x = rearrange(stft_repr, 'b f t c -> b t (f c)')
        freqs_per_bands_with_complex = []
        for i in range(len(DEFAULT_FREQS_PER_BANDS)):
            freqs_per_bands_with_complex.append(DEFAULT_FREQS_PER_BANDS[i] * 2 * 2)
        #freqs_per_bands_with_complex = tuple(2 * f * audio_channels for f in self.freqs_per_bands)
        
        x = BandSplit(
            dim=self.dim,
            freqs_per_bands_with_complex=freqs_per_bands_with_complex
        )(x)
        # transformer_kwargs = dict(
        #     dim=self.dim,
        #     heads=self.heads,
        #     dim_head=self.dim_head,
        #     attn_dropout=self.attn_dropout,
        #     ff_dropout=self.ff_dropout,
        #     norm_output=False
        # )
        for i in range(self.depth):
          x = rearrange(x, 'b t f d -> b f t d')
          x, ps = pack([x], '* t d')

          x = Transformer(depth=self.time_transformer_depth,
            name=f"time_transformer_{i}", 
            dim=self.dim,
            heads=self.heads,
            dim_head=self.dim_head,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            norm_output=False)(x,deterministic=deterministic)

          x, = unpack(x, ps, '* t d')
          x = rearrange(x, 'b f t d -> b t f d')
          x, ps = pack([x], '* f d')

          x = Transformer(depth=self.freq_transformer_depth,
            name=f"freq_transformer_{i}",
            dim=self.dim,
            heads=self.heads,
            dim_head=self.dim_head,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            norm_output=False)(x,deterministic=deterministic)

          x, = unpack(x, ps, '* f d')

        x = RMSNorm(self.dim)(x)
        out = []
        for _ in range(self.num_stems):
            res = MaskEstimator(
                dim=self.dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=self.mask_estimator_depth
            )(x)
            out.append(res)
        mask = jnp.stack(out,axis=1)
        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)
        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # complex number multiplication

        stft_repr = as_complex(stft_repr)
        mask = as_complex(mask)

        stft_repr = stft_repr * mask

        # istft
        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=audio_channels)
        t , recon_audio =util.istft(stft_repr,nfft=self.stft_n_fft,
            noverlap=self.stft_win_length-self.stft_hop_length,
            nperseg=self.stft_win_length,boundary=False,input_onesided=True)
        #recon_audio = as_real(recon_audio)
        #recon_audio = recon_audio.real
        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=audio_channels, n=self.num_stems)
        if self.num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        # if a target is passed in, calculate loss for learning

        #if not exists(target):
        return recon_audio


def as_complex(x):
   assert x.shape[-1] == 2
   return jax.lax.complex(x[...,0], x[...,1])
def as_real(x):
    if not jnp.issubdtype(x.dtype, jnp.complexfloating):
        return x

    xr = jnp.zeros(x.shape+(2,), dtype=x.real.dtype)
    xr = xr.at[...,0].set(x.real)
    xr = xr.at[...,1].set(x.imag)
    return xr

if __name__ =="__main__":
    test = BSRoformer(256,1)
    # init_arr = jnp.ones((1,2,16000))
    # rngs = {'params': jax.random.key(0), 'other_rng': jax.random.key(1)}
    # params_init = test.init(rngs,init_arr)
    # flatten_param = traverse_util.flatten_dict(params_init,sep='.')
    # print()
    from convert import load_params
    params = load_params()
    output = test.apply({"params":params},jnp.ones((1,2,16000)),deterministic=True)
    flatten_param = traverse_util.flatten_dict(params,sep='.')
    print(output)
    #print(output)
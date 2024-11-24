import flax
import jax
import torch
def load_params(hp):
    state_dict = torch.load("deverb_bs_roformer_8_256dim_8depth.ckpt")
    params = {

    }
    for i in range(62):
        params[f"BandSplit_0.RMSNorm_{i}.gamma"]=state_dict[f"band_split.to_features.{i}.0.gamma"]
        params[f"BandSplit_0.Dense_{i}.kernel"]=state_dict[f"band_split.to_features.{i}.1.weight"].transpose(0,1)
        params[f"BandSplit_0.Dense_{i}.bias"]=state_dict[f"band_split.to_features.{i}.1.bias"]
    for i in range(62):
        params[f"MaskEstimator_0.to_freqs_{i}.layers_0.layers_0.kernel"]=state_dict[f"mask_estimators.0.to_freqs.{i}.0.0.weight"].transpose(0,1)
        params[f"MaskEstimator_0.to_freqs_{i}.layers_0.layers_0.bias"]=state_dict[f"mask_estimators.0.to_freqs.{i}.0.0.bias"]
        params[f"MaskEstimator_0.to_freqs_{i}.layers_0.layers_2.kernel"]=state_dict[f"mask_estimators.0.to_freqs.{i}.0.2.weight"].transpose(0,1)
        params[f"MaskEstimator_0.to_freqs_{i}.layers_0.layers_2.bias"]=state_dict[f"mask_estimators.0.to_freqs.{i}.0.2.bias"]
    for i in range(hp.model.depth):
        #for j in range(2):

        params[f"time_transformer_{i}.layers_0_0.freqs"]=state_dict[f"layers.{i}.0.layers.0.0.rotary_embed.freqs"]
        params[f"time_transformer_{i}.layers_0_0.RMSNorm_0.gamma"]=state_dict[f"layers.{i}.0.layers.0.0.norm.gamma"]
        params[f"time_transformer_{i}.layers_0_0.to_qkv.kernel"]=state_dict[f"layers.{i}.0.layers.0.0.to_qkv.weight"].transpose(0,1)
        params[f"time_transformer_{i}.layers_0_0.to_gates.kernel"]=state_dict[f"layers.{i}.0.layers.0.0.to_gates.weight"].transpose(0,1)
        params[f"time_transformer_{i}.layers_0_0.to_gates.bias"]=state_dict[f"layers.{i}.0.layers.0.0.to_gates.bias"]
        params[f"time_transformer_{i}.layers_0_0.to_out.kernel"]=state_dict[f"layers.{i}.0.layers.0.0.to_out.0.weight"].transpose(0,1)
        params[f"time_transformer_{i}.layers_0_1.RMSNorm_0.gamma"]=state_dict[f"layers.{i}.0.layers.0.1.net.0.gamma"]
        params[f"time_transformer_{i}.layers_0_1.Dense_0.kernel"]=state_dict[f"layers.{i}.0.layers.0.1.net.1.weight"].transpose(0,1)
        params[f"time_transformer_{i}.layers_0_1.Dense_0.bias"]=state_dict[f"layers.{i}.0.layers.0.1.net.1.bias"]
        params[f"time_transformer_{i}.layers_0_1.Dense_1.kernel"]=state_dict[f"layers.{i}.0.layers.0.1.net.4.weight"].transpose(0,1)
        params[f"time_transformer_{i}.layers_0_1.Dense_1.bias"]=state_dict[f"layers.{i}.0.layers.0.1.net.4.bias"]

        params[f"freq_transformer_{i}.layers_0_0.freqs"]=state_dict[f"layers.{i}.1.layers.0.0.rotary_embed.freqs"]
        params[f"freq_transformer_{i}.layers_0_0.RMSNorm_0.gamma"]=state_dict[f"layers.{i}.1.layers.0.0.norm.gamma"]
        params[f"freq_transformer_{i}.layers_0_0.to_qkv.kernel"]=state_dict[f"layers.{i}.1.layers.0.0.to_qkv.weight"].transpose(0,1)
        params[f"freq_transformer_{i}.layers_0_0.to_gates.kernel"]=state_dict[f"layers.{i}.1.layers.0.0.to_gates.weight"].transpose(0,1)
        params[f"freq_transformer_{i}.layers_0_0.to_gates.bias"]=state_dict[f"layers.{i}.1.layers.0.0.to_gates.bias"]
        params[f"freq_transformer_{i}.layers_0_0.to_out.kernel"]=state_dict[f"layers.{i}.1.layers.0.0.to_out.0.weight"].transpose(0,1)
        params[f"freq_transformer_{i}.layers_0_1.RMSNorm_0.gamma"]=state_dict[f"layers.{i}.1.layers.0.1.net.0.gamma"]
        params[f"freq_transformer_{i}.layers_0_1.Dense_0.kernel"]=state_dict[f"layers.{i}.1.layers.0.1.net.1.weight"].transpose(0,1)
        params[f"freq_transformer_{i}.layers_0_1.Dense_0.bias"]=state_dict[f"layers.{i}.1.layers.0.1.net.1.bias"]
        params[f"freq_transformer_{i}.layers_0_1.Dense_1.kernel"]=state_dict[f"layers.{i}.1.layers.0.1.net.4.weight"].transpose(0,1)
        params[f"freq_transformer_{i}.layers_0_1.Dense_1.bias"]=state_dict[f"layers.{i}.1.layers.0.1.net.4.bias"]
    params["RMSNorm_0.gamma"] = state_dict["final_norm.gamma"]

    params = {k: v.numpy() for k, v in params.items()}
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    return params

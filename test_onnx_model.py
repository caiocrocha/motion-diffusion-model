#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import onnxruntime
import numpy as np
from sample.generate import load_model_and_diffusion, load_text
from utils.parser_util import generate_args
from utils import dist_util
from utils.fixseed import fixseed

def main(args=None, tolerance=1e-5):
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = generate_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    print('Loading text')
    texts, action_text, n_frames, max_frames, fps = load_text(args)
    print('Loading model and diffusion')
    model, diffusion, motion_shape, sample_fn = load_model_and_diffusion(args, data=None, n_frames=n_frames)
    sample_fn = diffusion.p_sample_loop_onnx

    scale = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    print('Encoding text')
    text_embed = model.encode_text(texts)
    enc_text, text_mask = text_embed

    print('Creating input data')
    x = torch.randn(*motion_shape).to(dist_util.dev())
    timesteps = torch.tensor([0]).to(dist_util.dev())
    sequence_length = enc_text.shape[0]
    print(f"Sequence length: {sequence_length}")
    batch_size = args.batch_size
    print(f"Batch size: {batch_size}")
    y = {
        'mask': torch.ones((batch_size, 1, 1, n_frames), dtype=torch.bool).to(dist_util.dev()),
        'lengths': torch.tensor([n_frames]).to(dist_util.dev()),
        'scale': scale.to(dist_util.dev()),
        'text_embed': (enc_text.to(dist_util.dev()), text_mask.to(dist_util.dev()))
    }

    dummy_input = (x, timesteps, y['mask'], y['lengths'], y['scale'], y['text_embed'][0], y['text_embed'][1])
    print(f"Dummy input shape: {[i.shape for i in dummy_input if isinstance(i, torch.Tensor)]}")
    input_onnx = {
        'batch_size': np.array([batch_size]),
        'njoints': np.array([263]),
        'nfeats': np.array([1]),
        'n_frames': np.array([n_frames]),
        'mask': y['mask'].cpu().numpy() if 'mask' in y else None,
        'scale': y['scale'].cpu().numpy() if 'scale' in y else None,
        'enc_text': y['text_embed'][0].cpu().numpy() if 'text_embed' in y else None,
        'text_mask': y['text_embed'][1].cpu().numpy() if 'text_embed' in y else None,
    }

    onnx_path = args.model_path.replace('.pt', '_onnx.onnx')
    print('Loading ONNX model')
    ort_session = onnxruntime.InferenceSession(onnx_path)
    # Load and run ONNX model
    print("\nONNX Model Expected Inputs:")
    for input_info in ort_session.get_inputs():
        print(f"Name: {input_info.name}")
        print(f"Shape: {input_info.shape}")
        print(f"Type: {input_info.type}")
        print("-" * 50)
    
    print('Running ONNX model')
    ort_output = ort_session.run(None, input_onnx)[0]

    torch_output = sample_fn(
        model,
        batch_size=1,
        njoints=263,
        nfeats=1,
        n_frames=120,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        mask=y['mask'],
        lengths=y['lengths'],
        text=None,
        tokens=None,
        scale=y['scale'],
        enc_text=y['text_embed'][0],
        text_mask=y['text_embed'][1],
        device=None,
        progress=True,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False
        )

    # Compare outputs
    print(f"ORT output shape: {ort_output.shape}")
    print(f"ORT output: {ort_output}")
    torch_output_np = torch_output.cpu().numpy()
    print(f"PyTorch output shape: {torch_output_np.shape}")
    print(f"PyTorch output: {torch_output_np}")
    max_diff = np.max(np.abs(torch_output_np - ort_output))
    mean_diff = np.mean(np.abs(torch_output_np - ort_output))
    
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    
    # Assert outputs are close within tolerance
    if max_diff > tolerance:
        raise AssertionError(f"Maximum difference ({max_diff}) exceeds tolerance ({tolerance})")

if __name__ == "__main__":
    main()
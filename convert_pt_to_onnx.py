#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from sample.generate import load_model_and_diffusion, load_text
from utils.parser_util import generate_args
from utils import dist_util
from utils.fixseed import fixseed
import numpy as np

def my_logical_not_function(g, input):
    """
    Implementation based on ONNX Not operator spec
    Reference: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Not
    """
    return g.op("Not", input)

# Register the custom operator
torch.onnx.register_custom_op_symbolic(
    "aten::logical_not",
    my_logical_not_function,
    1  # since opset 1
)

class MDMOnnxWrapper(nn.Module):
    def __init__(self, mdm_model, sample_fn, diffusion):
        super().__init__()
        self.model = mdm_model
        self.model.forward = self.model.forward_onnx
        if sample_fn == diffusion.p_sample_loop:
            self.sample_fn = diffusion.p_sample_loop_onnx
        else:
            raise ValueError("Unsupported sample function {}.".format(sample_fn))

    def forward(self, 
                batch_size=None,
                njoints=None,
                nfeats=None,
                n_frames=None, 
                mask=None,
                lengths=None,
                scale=None,
                enc_text=None,
                text_mask=None
        ):
        """Single denoising step that can be exported to ONNX
        """
        init_image = None
        sample = self.sample_fn(
            self.model,
            batch_size,
            njoints,
            nfeats,
            n_frames,
            clip_denoised=False,
            mask=mask,
            lengths=lengths,
            scale=scale,
            enc_text=enc_text,
            text_mask=text_mask,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=init_image,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        return sample

def export_mdm_to_onnx(model, sample_fn, diffusion, save_path, motion_shape, mask, lengths, scale, enc_text, text_mask):
    wrapper = MDMOnnxWrapper(model, sample_fn, diffusion)
    wrapper.eval()
    
    batch_size, njoints, nfeats, n_frames = motion_shape
    batch_size = torch.tensor([batch_size]).to(dist_util.dev())
    njoints = torch.tensor([njoints]).to(dist_util.dev())
    nfeats = torch.tensor([nfeats]).to(dist_util.dev())
    n_frames = torch.tensor([n_frames]).to(dist_util.dev())
    dummy_input = (batch_size, njoints, nfeats, n_frames, mask, lengths, scale, enc_text, text_mask)

    torch.onnx.export(
        wrapper,
        dummy_input,
        save_path,
        input_names=['batch_size', 'njoints', 'nfeats', 'n_frames', 
                     'mask', 'lengths', 'scale', 'enc_text', 'text_mask'],
        output_names=['output'],
        dynamic_axes={
            'mask': {0: 'batch_size', 3: 'n_frames'},
            'enc_text': {0: 'sequence_length'},
            'text_mask': {1: 'sequence_length'}
        },
        opset_version=17,
        verbose=False
    )

def main(args=None):
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = generate_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    texts, action_text, n_frames, max_frames, fps = load_text(args)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    model, diffusion, motion_shape, sample_fn = load_model_and_diffusion(args, data=None, n_frames=n_frames)

    scale = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    text_embed = model.encode_text(texts)
    enc_text, text_mask = text_embed

    x = torch.randn(*motion_shape).to(dist_util.dev())
    timesteps = torch.tensor([0]).to(dist_util.dev())
    batch_size = args.batch_size
    y = {
        'mask': torch.ones((batch_size, 1, 1, n_frames), dtype=torch.bool).to(dist_util.dev()),
        'lengths': torch.tensor([n_frames]).to(dist_util.dev()),
        'scale': scale.to(dist_util.dev()),
        'text_embed': (enc_text.to(dist_util.dev()), text_mask.to(dist_util.dev()))
    }

    dummy_input = (x, timesteps, y['mask'], y['lengths'], y['scale'], y['text_embed'][0], y['text_embed'][1])
    print(f"Dummy input shape: {[i.shape for i in dummy_input]}")

    export_path = args.model_path.replace('.pt', '_onnx.onnx')
    export_mdm_to_onnx(model, sample_fn, diffusion, export_path, motion_shape, 
                       y['mask'], y['lengths'], y['scale'], y['text_embed'][0], y['text_embed'][1])
    print(f"Exported model to {export_path}")

if __name__ == '__main__':
    main()
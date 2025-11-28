import torch
import torch.nn as nn
from einops import rearrange, reduce

from .components import COMPONENT
from .model import Model, Output


class GeMCL(Model):
    """Prototypical Network"""

    def __init__(self, config):
        super().__init__(config)
        x_enc_args = config['x_enc_args']
        x_enc_args['input_shape'] = config['x_shape']
        self.x_encoder = COMPONENT[config['x_encoder']](config, x_enc_args)
        assert config['output_type'] == 'class'
        self.ce = nn.CrossEntropyLoss(reduction='none')

        self.map = config['map']
        self.alpha = nn.Parameter(torch.tensor(100.), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(1000.), requires_grad=True)

    def forward(self, train_x, train_y, test_x, test_y, summarize, meta_split, sequential_class_by_class=True):
        batch, test_num = test_y.shape
        train_x_enc, test_x_enc = self.encode_x(train_x, test_x)

        # Configuration
        tasks = self.config['tasks']
        shots = self.config['train_shots']
        dim = train_x_enc.shape[-1]
        # 1. Reshape to separate tasks: [Batch, Tasks, Shots, Dim]
        train_x_reshaped = rearrange(train_x_enc, 'b (t s) d -> b t s d', t=tasks, s=shots)
        
        # 2. Initialize empty storage for all class parameters
        # We build the full model parameter set piece by piece
        prototypes = torch.zeros(batch, tasks, dim, device=train_x_enc.device)
        squared_diff = torch.zeros(batch, tasks, dim, device=train_x_enc.device)
        
        # 3. Sequential Loop (Class by Class)
        for t in range(tasks):
            # Extract all data for the current Task 't'
            # shape: [Batch, Shots, Dim]
            task_data = train_x_reshaped[:, t, :, :]
            
            # --- Learning Step for Task 't' ---
            # Since we have all shots for this class, we can use standard batch stats
            # This calculates the mean across the 'Shots' dimension
            task_mean = task_data.mean(dim=1) 
            
            # Calculate sum of squared differences for this class
            # (x - mean)^2
            # We unsqueeze mean to broadcast over shots: [Batch, 1, Dim]
            task_ssd = (task_data - task_mean.unsqueeze(1)).square().sum(dim=1)
            
            # --- Store Knowledge ---
            # Place the learned parameters into the global model storage
            prototypes[:, t, :] = task_mean
            squared_diff[:, t, :] = task_ssd

        # Train
        # prototypes = reduce(
        #     train_x_enc, 'b (t s) d -> b t d', 'mean', t=self.config['tasks'], s=self.config['train_shots'])
        # squared_diff = (
        #         rearrange(train_x_enc, 'b (t s) d -> b t s d', t=self.config['tasks']) -
        #         rearrange(prototypes, 'b t d -> b t 1 d')
        # ).square()
        # squared_diff = reduce(squared_diff, 'b t s d -> b t d', 'sum')

        alpha_prime = self.alpha + self.config['train_shots'] / 2
        beta_prime = self.beta + squared_diff / 2

        if self.map:
            var = beta_prime / (alpha_prime - 0.5)
        else:
            var = beta_prime / alpha_prime * (1 + 1 / self.config['train_shots'])

        # Test
        test_x_enc = rearrange(test_x_enc, 'b l h -> b l 1 h')
        prototypes = rearrange(prototypes, 'b t h -> b 1 t h')
        squared_diff = (test_x_enc - prototypes).square()  # b l t h
        var = rearrange(var, 'b t h -> b 1 t h')

        eps = 1e-8
        if self.map:
            nll = squared_diff / var + (var + eps).log()
            nll = reduce(nll, 'b l t h -> b l t', 'sum') / 2
        else:
            nll = (squared_diff / (var * alpha_prime * 2) + 1).log() * (alpha_prime + 0.5) + (var + eps).log()
            nll = reduce(nll, 'b l t h -> b l t', 'sum')

        logit = -nll
        loss = self.ce(rearrange(logit, 'b l t -> (b l) t'), rearrange(test_y, 'b l -> (b l)'))
        loss = reduce(loss, '(b n) -> b', 'mean', b=batch, n=test_num)

        output = Output()
        output[f'loss/meta_{meta_split}'] = loss
        if meta_split == 'test':
            output['predictions'] = logit.argmax(dim=-1)
        if not summarize:
            return output

        output.add_classification_summary(logit, test_y, meta_split)
        return output

from einops import reduce, rearrange
from torch import nn
import faiss
from models.components import COMPONENT, Mlp
from models.model import Output
from utils import OUTPUT_TYPE_TO_LOSS_FN
import random
import torch
import numpy as np

class REMIND(nn.Module):
    """Standard model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.replay_buffer = {}
        self.replay_buffer_size = config['replay_buffer_size']
        nbits = int(np.log2(config['codebook_size']))
        self.pq = faiss.ProductQuantizer(config['num_channels'], config['num_codebooks'], nbits) # placeholder
        enc_args = config['enc_args']
        enc_args['input_shape'] = config['x_shape']
        self.encoder = COMPONENT[config['encoder']](config, enc_args)

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        dec_args = config['dec_args']
        dec_args['output_shape'] = [config['tasks']] if config['output_type'] == 'class' else config['y_shape']
        self.decoder = COMPONENT[config['decoder']](config, dec_args)

        mlp_args = config['mlp_args']
        mlp_args['input_shape'] = self.encoder.output_shape
        mlp_args['output_shape'] = self.decoder.input_shape
        self.mlp = Mlp(config, mlp_args)

        # Define optimizer for MLP only
        self.optim = torch.optim.Adam(self.mlp.parameters(), **config['optim_args'])

        self.loss_fn = OUTPUT_TYPE_TO_LOSS_FN[config['output_type']]

    def forward(self, x, y, summarize, split):
        # assume the pq is trained and we have a buffer already.
        outputs=[]
        x = rearrange(x, 'b ... -> b 1 ...')
        y = rearrange(y, 'b ... -> b 1 ...')
        if split == 'train':
            for sample_x, sample_y in zip(x, y):
                # Encode the sample
                sample_encoded = self.encoder(sample_x)
                sample_encoded = rearrange(sample_encoded, '1 num_channels h w -> h w num_channels')
                # Compress using PQ
                sample_encoded = self.pq.compute_codes(rearrange(sample_encoded, 'h w num_channels -> (h w) num_channels'))
                sample_encoded = rearrange(sample_encoded, '(h w) num_codebooks -> h w num_codebooks', h=2, w=2)
                # sample 15 images from the replay buffer
                batch_codes, batch_labels = self.sample_replay_batch(sample_encoded, sample_y[0])
                batch_labels = rearrange(batch_labels, 'b ... -> b 1 ...')
                # reconstruct the batch
                batch_reconstructed = self.pq.decode(rearrange(batch_codes, 'b h w num_codebooks -> (b h w) num_codebooks'))
                batch_reconstructed = rearrange(batch_reconstructed, '(b h w) num_channels -> b num_channels h w', h=2, w=2, num_channels=self.config['num_channels'])
                # perform SGD on the MLP only
                logits = self.mlp(batch_reconstructed)
                loss = reduce(self.loss_fn(logits, batch_labels), 'b ... -> b', 'mean')

                # Backpropagate loss and update MLP parameters
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Add to replay buffer
                self.add_sample_to_replay_buffer(sample_encoded, sample_y[0])
                output = Output()
                output[f'loss/{split}'] = loss

                if not summarize:
                    outputs.append(output)
                output.add_classification_summary(logits, batch_labels, split)
                outputs.append(output)
        else:
            # if testing
            x_enc = self.encoder(x)
            x_enc = rearrange(x_enc, 'b num_channels h w -> b h w num_channels')
            x_enc = rearrange(x_enc, 'b h w num_channels -> (b h w) num_channels')
            x_enc = self.pq.compute_codes(x_enc)
            x_enc=self.pq.decode(x_enc)
            x_enc = rearrange(x_enc, '(b h w) num_channels -> b num_channels h w', h=2, w=2, num_channels=self.config['num_channels'])
            logits = self.mlp(x_enc)
            logits = self.decoder(logits)
            meta_loss = reduce(self.loss_fn(logits, y), 'b ... -> b', 'mean')

            output = Output()
            output[f'loss/{split}'] = meta_loss
            if not summarize:
                return output

            output.add_classification_summary(logits, y, split)
            outputs.append(output)
        return outputs
        
        # if testing
    
    def sample_replay_batch(self, current_code, current_label):
        """
        Sample codes from replay buffer and combine with current sample.
        
        Uniformly samples codes across all classes in the buffer and adds
        the current sample to create a training batch.
        
        Args:
            current_code: Current PQ-compressed feature (h, w, num_codebooks)
            current_label: Current class label
        
        Returns:
            batch_codes: Stacked codes of shape (num_replay+1, h, w, num_codebooks)
            batch_labels: Corresponding labels of shape (num_replay+1,)
        """
        # Flatten all codes and labels from buffer
        num_replay = self.config['batch_size'] - 1
        all_codes = []
        all_labels = []
        for label, codes in self.replay_buffer.items():
            all_codes.extend(codes)
            all_labels.extend([label] * len(codes))
        
        # Uniformly sample indices
        sample_indices = random.sample(range(len(all_codes)), num_replay)
        
        # Get sampled codes and labels
        sampled_codes = [all_codes[i] for i in sample_indices]
        sampled_labels = [all_labels[i] for i in sample_indices]
        
        # Add current sample at the beginning
        batch_codes = torch.stack([current_code] + sampled_codes)
        batch_labels = torch.tensor([current_label] + sampled_labels)
        
        return batch_codes, batch_labels
    
    def add_sample_to_replay_buffer(self, sample_encoded, sample_y):
        # Source - https://stackoverflow.com/a
        # Posted by zwol, modified by community. See post 'Timeline' for change history
        # Retrieved 2025-11-12, License - CC BY-SA 4.0

        current_size = sum(len(sample) for sample in self.replay_buffer.values())

        if int(sample_y) not in self.replay_buffer:
            self.replay_buffer[int(sample_y)] = []

        if current_size >= self.replay_buffer_size:
            max_samples = max(len(samples) for samples in self.replay_buffer.values())
            max_classes = [class_ for class_, samples in self.replay_buffer.items() 
                       if len(samples) == max_samples]
            selected_class = random.choice(max_classes)
            idx = random.randrange(len(self.replay_buffer[selected_class]))
            self.replay_buffer[selected_class].pop(idx)

        self.replay_buffer[int(sample_y)].append(sample_encoded)


    def reconstruct_from_replay_buffer(self, codes_x):
        train_x_reconstructed = self.pq.decode(codes_x) # * The shape should be (batch_size, h, w, num_channels)
        train_x_reconstructed = rearrange(train_x_reconstructed, 'b h w num_channels -> b num_channels h w')
        return train_x_reconstructed
        

    def train_pq(self, train_x_initial):

        train_x_initial = rearrange(train_x_initial, 'b ... -> b 1 ...')
        logit = self.encoder(train_x_initial)
        logit = rearrange(logit, 'b num_channels h w -> b h w num_channels')
        logit = rearrange(logit, 'b h w num_channels -> (b h w) num_channels') # flatten (images*2*2, num_channels)
        self.pq.train(logit)
        # ? Should I do this by batches or all at once? The original paper does it in batches.
        self.codes = self.pq.compute_codes(logit)
        self.codes = rearrange(self.codes, '(b h w) num_codebooks -> b h w num_codebooks', h=2, w=2)

    def initialise_buffer_with_initial_data(self, train_x_initial, train_y_initial):
        """
        Initialise the replay buffer with base initialisation data.
        
        Trains the Product Quantiser on initial features and populates the buffer.
        Buffer structure: {label: [codes]} where each code is a PQ-compressed
        feature map of shape (h, w, num_codebooks). Each label maintains a list
        of all stored codes for that class.
        
        Args:
            train_x_initial: Initial training images
            train_y_initial: Corresponding class labels
        """
        self.train_pq(train_x_initial)

        for label, code in zip(train_y_initial, self.codes):
            self.replay_buffer.setdefault(int(label), []).append(code)

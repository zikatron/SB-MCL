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
        self.pq = faiss.ProductQuantizer(512, config['num_codebooks'], nbits) # placeholder
        enc_args = config['enc_args']
        enc_args['input_shape'] = config['x_shape']
        self.encoder = COMPONENT[config['encoder']](config, enc_args)


        dec_args = config['dec_args']
        dec_args['output_shape'] = [config['tasks']] if config['output_type'] == 'class' else config['y_shape']
        self.decoder = COMPONENT[config['decoder']](config, dec_args)

        mlp_args = config['mlp_args']
        mlp_args['input_shape'] = self.encoder.output_shape
        mlp_args['output_shape'] = self.decoder.input_shape
        self.mlp = Mlp(config, mlp_args)

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.eval()  # Set encoder to eval mode

        # # Freeze all layers in the MLP
        # for param in self.mlp.parameters():
        #     param.requires_grad = False

        # # Unfreeze the last layer
        # for param in self.mlp[-1].parameters():  # Assuming self.mlp is a Sequential model
        #     param.requires_grad = True

        # Define optimizer for MLP only
        self.optim = torch.optim.SGD(self.mlp.parameters(), **config['optim_args'])
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=10, gamma=0.1)

        self.loss_fn = OUTPUT_TYPE_TO_LOSS_FN[config['output_type']]

    def forward(self, x, y, summarize, split):
        # assume the pq is trained and we have a buffer already.
        outputs=[]
        x = rearrange(x, 'b ... -> b 1 ...')
        y = rearrange(y, 'b ... -> b 1 ...')
        x = self.encoder(x)  # GPU torch tensor

        if split == 'train':
            for sample_encoded, sample_y in zip(x, y):
                # NEED: Convert to CPU numpy for PQ
                sample_encoded_np = sample_encoded.cpu().numpy()
                codes = self.pq.compute_codes(sample_encoded_np) 
                
                # codes is numpy, pass to sample_replay_batch
                batch_codes, batch_labels = self.sample_replay_batch(codes, sample_y[0].item())
                batch_labels = rearrange(batch_labels, 'b ... -> b 1 ...')
                
                # batch_codes is numpy (from sample_replay_batch)
                batch_reconstructed = self.pq.decode(
                    rearrange(batch_codes, 'b 1 num_codebooks -> (b 1) num_codebooks')
                )  # Returns numpy
                
                # NEED: Convert to GPU torch for MLP
                batch_reconstructed = torch.from_numpy(batch_reconstructed).cuda()
                batch_reconstructed = rearrange(
                    batch_reconstructed, 
                    '(b 1) num_channels -> b 1 num_channels'
                )
                batch_labels = torch.from_numpy(batch_labels).cuda()
                logits = self.mlp(batch_reconstructed)
                logits = self.decoder(logits)

                loss = reduce(self.loss_fn(logits, batch_labels), 'b ... -> ()', 'mean')

                # Backpropagate loss and update MLP parameters
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Add to replay buffer
                self.add_sample_to_replay_buffer(codes, sample_y[0].item())
                output = Output()
                output[f'loss/{split}'] = loss

                if not summarize:
                    outputs.append(output)
                output.add_topk_classification_summary(logits, batch_labels, split, k=1)
                outputs.append(output)
        else:          
            # NEED: Convert to CPU numpy
            x_enc = x.cpu().numpy()
            
            x_enc = rearrange(x_enc, 'b 1 num_channels -> (b 1) num_channels')
            x_enc = self.pq.compute_codes(x_enc)
            x_enc = self.pq.decode(x_enc)
            
            # NEED: Convert back to GPU torch
            x_enc = torch.from_numpy(x_enc).cuda()
            
            x_enc = rearrange(x_enc, '(b 1) num_channels -> b 1 num_channels')
            logits = self.mlp(x_enc)
            logits = self.decoder(logits)
            loss = reduce(self.loss_fn(logits, y), 'b ... -> b', 'mean')

            output = Output()
            output[f'loss/{split}'] = loss
            output['predictions'] = logits.argmax(dim=-1).squeeze()
            output['labels'] = y.squeeze(-1)
            if not summarize:
                return output

            output.add_topk_classification_summary(logits, y, split, k=1)
            outputs.append(output)
        # current_size = sum(len(sample) for sample in self.replay_buffer.values())
        # print("Current replay buffer size:", current_size)
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
            all_codes.extend(codes)  # All numpy arrays
            all_labels.extend([label] * len(codes))
        
        sample_indices = random.sample(range(len(all_codes)), num_replay)
        sampled_codes = [all_codes[i] for i in sample_indices]
        sampled_labels = [all_labels[i] for i in sample_indices]
        
        # Stack numpy arrays
        batch_codes = np.stack([current_code] + sampled_codes)  #  Change to np.stack
        batch_labels = np.array([current_label] + sampled_labels)  # Change to np.array
        
        return batch_codes, batch_labels  # Both numpy
    
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


    # def reconstruct_from_replay_buffer(self, codes_x):
    #     train_x_reconstructed = self.pq.decode(codes_x) # * The shape should be (batch_size, h, w, num_channels)
    #     train_x_reconstructed = rearrange(train_x_reconstructed, 'b h w num_channels -> b num_channels h w')
    #     return train_x_reconstructed
        

    def train_pq(self, train_x_initial):
        
        train_x_initial = rearrange(train_x_initial, 'b ... -> b 1 ...')
        logit = self.encoder(train_x_initial)  # GPU torch tensor
        print("Encoder output shape:", logit.shape)
        # NEED: Convert to CPU numpy for PQ
        logit = logit.cpu().numpy() 
        # the shape is (b,1,512) there's a linear layer in the cnn encoder
        logit = rearrange(logit, 'b 1 output_shape -> (b 1) output_shape', b=5000, output_shape=self.config['num_channels'])
        self.pq.train(logit)  # Now CPU numpy 
        codes = self.pq.compute_codes(logit)  # Returns numpy
        # print("Initial codes shape:", codes.shape)
        # print("Codebook size:", self.pq.code_size)
        # print("Num codebooks:", self.pq.M)
        # print("Dimensionality:", self.pq.d)
        # print("Bits per codebook:", self.pq.nbits)
        codes = rearrange(codes, '(b 1) num_codebooks -> b 1 num_codebooks', b=5000)
        return codes

    def initialise_buffer_with_initial_data(self, train_x_initial, train_y_initial):
        """
        Initialise the replay buffer with base initialisation data.
        
        Trains the Product Quantiser on initial features and populates the buffer.
        Buffer structure: {label: [codes]} where each code is a PQ-compressed
        feature map. Each label maintains a list
        of all stored codes for that class.
        
        Args:
            train_x_initial: Initial training images
            train_y_initial: Corresponding class labels
        """
        codes = self.train_pq(train_x_initial)
        for label, code in zip(train_y_initial, codes):
            self.replay_buffer.setdefault(int(label), []).append(code)

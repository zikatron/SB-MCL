from einops import reduce, rearrange
from torch import nn
import faiss
from models.components import COMPONENT, Mlp
from models.model import Output
from utils import OUTPUT_TYPE_TO_LOSS_FN


class REMIND(nn.Module):
    """Standard model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.replay_buffer = {}
        self.pq = faiss.ProductQuantizer() # placeholder
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

        self.loss_fn = OUTPUT_TYPE_TO_LOSS_FN[config['output_type']]

    def forward(self, train_x, train_y, test_x, test_y, summarize, split, pq, replay_buffer):
        # assume the pq is trained and we have a buffer already.
        train_x = rearrange(train_x, 'b ... -> b 1 ...')
        train_y = rearrange(train_y, 'b ... -> b 1 ...')
        logit = self.encoder(train_x)
        # logit = self.mlp(logit)
        # logit = self.decoder(logit)
        # loss = reduce(self.loss_fn(logit, train_y), 'b ... -> b', 'mean')
        # output = Output()
        # output[f'loss/{split}'] = loss
        # if not summarize:
        #     return output

        # if self.config['output_type'] == 'class':
        #     output.add_classification_summary(logit, train_y, split)
        # elif self.config['output_type'] == 'image':
        #     output.add_image_comparison_summary(
        #         rearrange(train_x, 'b 1 ... -> 1 b ...'),
        #         rearrange(train_y, 'b 1 ... -> 1 b ...'),
        #         rearrange(train_x, 'b 1 ... -> 1 b ...'),
        #         rearrange(logit, 'b 1 ... -> 1 b ...'),
        #         key=f'completion/{split}')
        # return output
    
    def code_encode(self, logit, train_y):
        """
        This will encode the representations and select f other samples from the replay buffer
        to make a batch. Then we proceed to run SGD on that batch and make updates to just the mlp.
        """
        pq.code(logit)

        # here i will select f other samples from the replay buffer
        # combine them with the current sample to form a batch
        # reconstruct the batch
        # perform SGD and update the MLP only.
        # add sample to the replay buffer.

        pass

    def online_training(self, samples_encoded, train_y):
        samples_decoded = self.pq.decode(samples_encoded)
        logits = self.mlp(samples_decoded)
        return logits
    
    def add_simple_to_replay_buffer(self, sample_encoded, sample_y):

        pass

    def reconstruct_from_replay_buffer(self):
        pass

    def intialize_replay_buffer(self, train_x, train_y):
        train_x = rearrange(train_x, 'b ... -> b 1 ...')
        train_y = rearrange(train_y, 'b ... -> b 1 ...')
        logit = self.encoder(train_x)
        train_x_encoded = pq.encode(logit)
        # add this to the replay buffer
        
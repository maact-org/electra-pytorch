import torch
from torch import nn
from transformers import AlbertConfig, AlbertModel

class AdaptedDiscriminator(nn.Module):
    '''
    Class that adapts an Albert general language model to the task of discriminating tokens depending on weather they
    come from an original distribution or from a generator
    '''

    def __init__(self,discriminator_configuration:AlbertConfig):
        super(AdaptedDiscriminator, self).__init__()
        self.config = discriminator_configuration
        self.albert_model = AlbertModel(discriminator_configuration)
        self.electra = self.albert_model
        self.linear_layer = nn.Linear(discriminator_configuration.hidden_size, 1)
        # self.discriminator = nn.Sigmoid()

    def get_lm(self):
        '''
        get general language model
        :return: the general Albert language model
        '''
        return self.albert_model

    def save_lm(self, name):
        directory = SAVE_DIRECTORY.format(name)
        self.config.save_pretrained(directory)
        torch.save(self.albert_model.state_dict(), directory + 'pytorch_model.bin')

    def forward(self, masked_input, **kwargs):
        """
        Pytorch forward function. (for more info read the pytorch documentation)
        :param masked_input: A tensor with shape (batch_size,max_position_embedding) containing sequences of token ids
        created by TOKENIZER with some of them masked
        :return: returns a Tensor with shape (batch_size,max_position_embedding) containing the probability that
        each given token comes from the generator
        """
        albert_out = self.albert_model(masked_input, **kwargs)[0]
        linear_out = self.linear_layer(albert_out).squeeze(dim=-1)
        return [linear_out]

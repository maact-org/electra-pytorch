import torch
from torch import nn
from transformers import AlbertConfig, AlbertModel

from alectra.settings import DISCRIMINATOR_HIDDEN_SIZE, DISCRIMINATOR_NUM_ATTENTION_HEADS, \
    DISCRIMINATOR_INTERMEDIATE_SIZE, VOCAB_SIZE, MAX_LEN, SAVE_DIRECTORY


class AdaptedDiscriminator(nn.Module):
    '''
    Class that adapts an Albert general language model to the task of discriminating tokens depending on weather they
    come from an original distribution or from a generator
    '''

    def __init__(self,
                 hidden_size=DISCRIMINATOR_HIDDEN_SIZE,
                 num_attention_heads=DISCRIMINATOR_NUM_ATTENTION_HEADS,
                 intermediate_size=DISCRIMINATOR_INTERMEDIATE_SIZE,
                 vocab_size=VOCAB_SIZE,
                 max_position_embeddings=MAX_LEN,
                 ):
        '''
        Class initializer
        :param hidden_size: Size of the output embeddings
        :param num_attention_heads: Number of attention heads
        :param intermediate_size: Intermediate size
        :param vocab_size: Size of the vocabulary of tokens, it must be equal to the tokenizer vocab_size
        :param max_position_embeddings: Max length for the ids sequences given by the tokenizer
        '''
        super(AdaptedDiscriminator, self).__init__()
        discriminator_configuration = AlbertConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings
        )
        self.config = discriminator_configuration
        self.albert_model = AlbertModel(discriminator_configuration)
        self.electra = self.albert_model
        self.linear_layer = nn.Linear(hidden_size, 1)
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

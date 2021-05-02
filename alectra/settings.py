import torch

# Tokenizer to be used in the training process
TOKENIZER_PATH = 'data/Tokenizer.model'

# Data set used for training
DATA_SET_PATH = 'data/wiki_pt.csv'

# SETTING: Max length for the ids sequences given by the tokenizer
MAX_LEN = 512

# SETTING: Batch size
BATCH_SIZE = 4

# SETTING: A string specifying the device to be used usually cuda:0 or cpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# SETTING: path to save the model in(includes extension)
SAVE_DIRECTORY = 'data/{}/'

# SETTING: Number of epochs for the training
EPOCHS = 500

# Default vocab size for the generator and discriminator, it must be equal to the tokenizer vocab_size
VOCAB_SIZE = 30522

# Default size of the output embeddings of the generator
GENERATOR_HIDDEN_SIZE = 192

# Default number of attention heads for the generator
GENERATOR_NUM_ATTENTION_HEADS = 12

# Default intermediate size for the generator
GENERATOR_INTERMEDIATE_SIZE = 768

# Default size of the output embeddings of the discriminator
DISCRIMINATOR_HIDDEN_SIZE = 768

# Default number of attention heads for the discriminator
DISCRIMINATOR_NUM_ATTENTION_HEADS = 12

# Default intermediate size gor the discriminator
DISCRIMINATOR_INTERMEDIATE_SIZE = 3072

# Id of mask token
MASK_TOKEN_ID = 3

# Id of padding token
PAD_TOKEN_ID = 0

# Porbability that a given id will be masked in training
MASK_PROB = 0.15

# saving frequency
SAVING_FREQUENCY = 1000

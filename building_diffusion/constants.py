"""
Fixed values of the Building-GAN dataset and of the trained models.
They are baked into the processed data and the checkpoints: changing them breaks both.
"""

# Program types: a voxel or program node is one of 6 programs; a voxel can also be unassigned
PROGRAM_CLASS_NAMES = ('Lobby', 'Restroom', 'Stairs', 'Elevator', 'Office', 'Mechanical')
NUM_PROGRAM_CLASSES = 6
UNASSIGNED_LABEL = 6  # Voxel label index of unassigned voxels
NUM_VOXEL_CLASSES = 7  # 6 programs + unassigned
VOXEL_CLASS_NAMES = PROGRAM_CLASS_NAMES + ('Unassigned',)

# Floors
NUM_FLOORS = 11  # Size of the floor one-hot encoding (program features, SSL floor classifier, FGW node features)
MAX_FLOORS_POS_ENC = 20  # Rows of the voxel floor positional encoding table

# Input features
PROGRAM_FEATURES = 19  # SSL input: 6 program class + 11 floor + node TPR + building FAR
VOXEL_FEATURES = 14  # Diffusion input: 7 voxel features (dimension x3, coordinate x3, weight) + 7 noisy label
FAR_SCALE = 0.1  # The building FAR is multiplied by this in the program features

# Voxel class frequencies of the dataset (6 programs + unassigned): the reverse diffusion starts from them
VOXEL_CLASS_PRIOR = [0.1324, 0.0637, 0.0278, 0.0495, 0.3826, 0.0113, 0.3327]

# FGW distance: alpha = 0 is replaced by this value, as POT divides by alpha (the change of the distance is negligible)
FGW_MIN_ALPHA = 1e-8

# Average number of nodes per graph of the Building-GAN dataset, used by the noise schedule
AVERAGE_NUM_NODES = 396

# Frozen SSL encoder inside BuildingDiffusion: must match the trained SSL checkpoints
SSL_LAYER_NAME = "GATv2"
SSL_NUM_LAYERS = 5  # Hidden layers, all of the SSL embedding size
SSL_ENCODING = 64
SSL_OUT_CHANNELS = 64
SSL_HEADS = 8
SSL_CONCAT = True
SSL_DROPOUT = 0.5

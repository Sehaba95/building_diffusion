import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from building_diffusion.constants import NUM_VOXEL_CLASSES, VOXEL_CLASS_PRIOR


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb, device):
    """ 
    Code taken from DiGress paper
        M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
    """
    # Flatten feature tensors
    # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
    # bs x (n ** 2) x d
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32).to(device)            # bs x N x dt

    left_term = X_t @ Qt.transpose(-1, -2)    # bs, N, 1, d_t-1
    numerator = (left_term.unsqueeze(2) * Qsb.unsqueeze(1))         # bs, N, d0, d_t-1

    # Compute denominator
    denominator = (Qtb @ X_t.transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)       # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator

    return out

def sample_p_zs_given_zt(s, t, X_t, X_predicted, noise_scheduler, device):
    """
    Code taken from DiGress paper
    Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well.
        s: timestep -1 
        t: timestep 
        X_t: Voxel Graph at Timestep t.
        X_predicted: Voxel Graph at Timestep s (t-1)."""
    bs, n, dxs = X_t.shape

    beta_t = noise_scheduler(t_normalized=t).to(device)  # (bs, 1)
    alpha_s_bar = noise_scheduler.get_alpha_bar(t_normalized=s)
    alpha_t_bar = noise_scheduler.get_alpha_bar(t_normalized=t)

    # Retrieve transitions matrix
    Qtb = noise_scheduler.get_Qt_bar(alpha_t_bar).to(device)
    Qsb = noise_scheduler.get_Qt_bar(alpha_s_bar).to(device)
    Qt = noise_scheduler.get_Qt(beta_t).to(device)

    # Normalize predictions
    pred_X = F.softmax(X_predicted, dim=-1)               # bs, n, d0

    p_s_and_t_given_0_X = compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                       Qt=Qt,
                                                                       Qsb=Qsb,
                                                                       Qtb=Qtb,
                                                                       device=device)

    # Compute weighted probabilities
    unnormalized_prob_X = (pred_X.unsqueeze(-1) * p_s_and_t_given_0_X).sum(dim=2)         # bs, n, d_t-1
    unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5

    # Normalize probabilities
    prob_X = (unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True))  # bs, n, d_t-1

    assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()

    sampled_s = noise_scheduler.sample_discrete_features(prob_X)

    X_s = F.one_hot(sampled_s, num_classes=NUM_VOXEL_CLASSES).float()

    assert (X_t.shape == X_s.shape)

    return X_s

def sample_from_distribution(num_samples, probabilities=VOXEL_CLASS_PRIOR):
    """
    Samples values from a given probability distribution and returns one-hot encoded vectors.

    Parameters:
    probabilities: class probabilities (default: VOXEL_CLASS_PRIOR, calculated from the dataset).
    num_samples (int): Number of random node label values to generate.

    Returns:
    torch.Tensor: A tensor of one-hot encoded vectors.
    """
    # Convert probabilities to a torch tensor
    probabilities = torch.tensor(probabilities, dtype=torch.float32)
    
    # Create a categorical distribution
    m = Categorical(probabilities)
    
    # Sample from the distribution
    samples = m.sample((num_samples,))
    
    # Convert the samples to one-hot encoded vectors
    one_hot_vectors = torch.nn.functional.one_hot(samples, num_classes=len(probabilities)).to(dtype=torch.float32)
    
    return one_hot_vectors

def reverse_timesteps(timesteps, device):
    """Timesteps T-1, ..., 0 (shape T x 1 x 1), with their normalised values t / T and (t - 1) / T."""
    # Precompute the timesteps from 999 or 499 to 0 to avoid creating them in each iteration
    timestep_tensor = torch.arange(timesteps-1, -1, -1).unsqueeze(1).unsqueeze(1).to(device)
    norm_timesteps = (timestep_tensor / timesteps).to(device)
    norm_timesteps_minus_one = ((timestep_tensor - 1) / timesteps).to(device)
    return timestep_tensor, norm_timesteps, norm_timesteps_minus_one

def reverse_diffusion(model, data, program_feat, floor_pe, noise_scheduler, schedule, device, on_step=None):
    """
    Generate voxel labels for a batch by running the reverse diffusion process.
    Start from labels drawn from the dataset class distribution, and at every timestep predict the
    clean graph and sample the less noisy graph at t-1 (DiGress sampling).
        schedule: output of reverse_timesteps()
        on_step: optional callback(timestep, labels) called after every step with the N x 7 one-hot labels
    Returns the final one-hot labels (N x 7) and the model logits of the last step (N x 7).
    """
    timestep_tensor, norm_timesteps, norm_timesteps_minus_one = schedule

    # Initialize the input random graphs following the dataset distribution
    g_t = sample_from_distribution(num_samples=data.voxel_label.size(0))
    g_t = g_t.unsqueeze(0).to(device)

    # Unused value, but the draw advances the global RNG: removing it would change every later sample
    torch.randint(0, 24001, (1,))

    # Iterate over timesteps and predict less noisy voxel graph
    for timestep in range(timestep_tensor.size(0)):
        # 14 Features : Dimension, coordinates, weight, g_t for noisy labels (6 classes + assigned status)
        node_features = torch.cat((data.voxel_feature, g_t.squeeze(0)), dim=1).clone().to(device)

        # Predict the less noisy building
        voxel_predictions = model(node_features, data.voxel_edge, data.voxel_feature_batch, timestep_tensor[timestep], program_feat, floor_pe=floor_pe)

        # Sample Voxel Graph at Timestep t-1
        g_t = sample_p_zs_given_zt(norm_timesteps_minus_one[timestep], norm_timesteps[timestep], X_t=g_t, X_predicted=voxel_predictions, noise_scheduler=noise_scheduler, device=device)

        if on_step is not None:
            on_step(timestep_tensor[timestep].item(), g_t.squeeze(0))

    return g_t.squeeze(0), voxel_predictions

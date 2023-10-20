import torch
from torch import autograd
from torch import nn
import numpy as np
import random
cuda = True if torch.cuda.is_available() else False

def get_noise(batch_size, noise_size):
    '''
    Function for creating noise vectors: Given the dimensions (batch_size, noise_size),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    if cuda:
        return torch.randn((batch_size, noise_size), device="cuda")
    else:
        return torch.randn((batch_size, noise_size))

def get_energy(batch_size, min_energy = 0, max_energy = 1):
    return torch.rand((batch_size, 1)) * (max_energy - min_energy) + min_energy

def generate_fake_data(gen, batch_size):
    '''
    The generator model is used to produce the average PDF, which means the model produce the same image every time. There are statistical fluctuations which depends on the number of shots.
    Such fluctuations have negligible impact on the results. To save the training time, we only produce one image for each batch.
    '''
    noise = get_noise(batch_size, gen.input_size)
    images = gen(noise)
    return images

def generate_validation_data(gen, batch_size):
    '''
    The validation data is used in the comparison between fake data and real data, where statistical fluctuations matter. To reduce the fluctuations, we produce 1000 images.
    '''
    gen.eval()
    with torch.inference_mode():
        noise = get_noise(batch_size, gen.input_size)
        images = gen(noise)
    gen.train()
    return images

def weight_clip(model, limit = 0.01):
    for para in model.parameters():
        para.data.clamp_(-limit, limit)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    if cuda:
        alpha = torch.rand((real_samples.size(0), 1), device="cuda")
    else:
        alpha = torch.rand((real_samples.size(0), 1))
        
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    if cuda:
        fake = autograd.Variable(torch.ones((real_samples.size(0), 1), device="cuda"), requires_grad=False)
    else:
        fake = autograd.Variable(torch.ones((real_samples.size(0), 1)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def cal_disc_ce_loss(disc, real_data, fake_data):
    criterion = nn.BCEWithLogitsLoss()

    disc_real_pred = disc(real_data)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))

    disc_fake_pred = disc(fake_data.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

    disc_loss = (disc_real_loss + disc_fake_loss)/2

    return disc_loss, disc_real_loss, disc_fake_loss

def cal_gen_ce_loss(disc, fake_data):
    criterion = nn.BCEWithLogitsLoss()

    disc_fake_pred = disc(fake_data)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

    return gen_loss

def cal_disc_W_loss(disc, real_data, fake_data, lambda_gp = 10):
    disc_real_pred = disc(real_data)
    disc_real_loss = -disc_real_pred.mean()
    
    disc_fake_pred = disc(fake_data)
    disc_fake_loss = disc_fake_pred.mean()

    disc_loss = disc_real_loss + disc_fake_loss
    if lambda_gp > 0.001:
        gradient_penalty = compute_gradient_penalty(disc, real_data, fake_data)
        disc_loss +=  lambda_gp * gradient_penalty

    return disc_loss, disc_real_loss, disc_fake_loss, lambda_gp * gradient_penalty

def cal_gen_W_loss(disc, fake_data):
    disc_fake_pred = disc(fake_data)
    gen_loss = - disc_fake_pred.mean()

    return gen_loss

def checkpoint(gen, gen_optim, gen_scheduler, disc = None, disc_optim = None, disc_scheduler = None, path = ''):
    torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    rng_state = random.getstate()
    
    if disc:
        state_dict = {
        'gen': gen.state_dict(),
        'gen_optim': gen_optim.state_dict(),
        'gen_scheduler': gen_scheduler.state_dict(),
        'disc': disc.state_dict(),
        'disc_optim': disc_optim.state_dict(),
        'disc_scheduler':disc_scheduler.state_dict(),
        'torch_rng_state': torch_rng_state,
        'np_rng_state': np_rng_state,
        'rng_state': rng_state,
        }
    else:
        state_dict = {
        'gen': gen.state_dict(),
        'gen_optim': gen_optim.state_dict(),
        'gen_scheduler': gen_scheduler.state_dict(),
        'torch_rng_state': torch_rng_state,
        'np_rng_state': np_rng_state,
        'rng_state': rng_state,
        }        

    torch.save(state_dict, path)

def resume(gen, gen_optim, gen_scheduler, disc = None, disc_optim = None, disc_scheduler = None, path = ''):
    checkpoint = torch.load(path)
    gen.load_state_dict(checkpoint['gen'])
    gen_optim.load_state_dict(checkpoint['gen_optim'])
    gen_scheduler.load_state_dict(checkpoint['gen_scheduler'])

    if disc:
        disc.load_state_dict(checkpoint['disc'])
        disc_optim.load_state_dict(checkpoint['disc_optim'])
        disc_scheduler.load_state_dict(checkpoint['disc_scheduler'])

    torch.set_rng_state(checkpoint['torch_rng_state'])
    np.random.set_state(checkpoint['np_rng_state'])
    random.setstate(checkpoint['rng_state'])


def describe_backend(backend):
    config = backend.configuration()
    print(f'Information about backend {config.backend_name} with version {config.backend_version}:')
    print(f'number of qubits: {config.num_qubits}')
    print(f'basis gate: {config.basis_gates}')
    print(f'coupling map: {config.coupling_map}\n')

    properties = backend.properties()

    print('Information of the single qubit:')
    for qubit in range(config.n_qubits):
        print(f'Qubit {qubit} has:')
        print(f'    - t1: {properties.t1(qubit)}')
        print(f'    - t2: {properties.t2(qubit)}')
        print(f"    - x error: {properties.gate_error('x', qubit)*100:.3f}%")
        print(f'    - readout error(\033[94msymmetrized\033[0m): {properties.readout_error(qubit)*100:.1f}%')

    print('\nInformation of the qubit pair:')
    for qubit_pair in config.coupling_map:
        print(f'Qubit {qubit_pair} has:')
        print(f"    - cx error: {properties.gate_error('cx', qubit_pair)*100:.1f}%")

def describe_noise_model(noise_model):
    print('The noise model: \n')
    dict = noise_model.to_dict()
    for error in dict['errors']:
        if error['operations'][0] != 'measure':
            continue

        if 'gate_qubits' in error:
            qubit = error['gate_qubits'][0][0]
            print(f'Qubit {qubit} has')
        else:
            print('All qubit has')

        p00 = error['probabilities'][0][0]
        p11 = error['probabilities'][1][1]
        print(f'    - |0> readout fidelity: {p00*100:.2f}%')
        print(f'    - |1> readout fidelity: {p11*100:.2f}%')

    for error in dict['errors']:
        if error['operations'][0] != 'cz':
            continue

        if 'gate_qubits' in error:
            coupler = error['gate_qubits'][0]
            print(f'Coupler {coupler} has')
        else:
            print('All coupler has')

        error_rate = 1 - error['probabilities'][0] + error['probabilities'][1]
        print(f'    - cz gate error: {error_rate*100:.2f}%')

    for error in dict['errors']:
        if error['operations'][0] != 'cx':
            continue

        if 'gate_qubits' in error:
            coupler = error['gate_qubits'][0]
            print(f'Coupler {coupler} has')
        else:
            print('All coupler has')

        error_rate = 1 - error['probabilities'][0] + error['probabilities'][1]
        print(f'    - cx gate error: {error_rate*100:.2f}%')


def plot_coupling_map(coupling_map):
    from qiskit.transpiler import CouplingMap
    cmap = CouplingMap(coupling_map)
    return cmap.draw()
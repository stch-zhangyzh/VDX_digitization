import torch
from torch import nn
from torchinfo import summary
from torch.autograd import Function


class StepFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.heaviside(input, torch.zeros_like(input))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        return grad_input

class Step(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return StepFunction.apply(input)


class DensityNN(nn.Module):
    '''
    NN class with a density as output.
    '''
    
    def __init__(self, input_size, hidden_layers_size, output_size=1, batch_norm=True, last_activation=None):
        '''
        Initialize the model, which is constructed by the linear blocks
        Parameters:
            input_size: the size of the latent vector, a scalar
            hidden_layers_size: the size of the hidden layers, a vector
            output_size: the size of the generated image, a scalar. If the output_size equals to 1, then
                         the output layer will be a linear layer. Otherwise, it will be a linear layer 
                         followed by a Sigmoid layer.
        '''
        super(DensityNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_norm = batch_norm

        # hidden layers
        self.net = self._linear_block(input_size, hidden_layers_size[0])
        for layer in range(1, len(hidden_layers_size)):
            _hidden_layer = self._linear_block(hidden_layers_size[layer-1],
                                               hidden_layers_size[layer])
            self.net = nn.Sequential(*(self.net.children()), *(_hidden_layer.children()))

        # final layer
        self.net = nn.Sequential(*(self.net.children()),
                                nn.Linear(hidden_layers_size[-1], output_size))
        if last_activation == 'sigmoid':
            self.net = nn.Sequential(*(self.net.children()),
                                     nn.Sigmoid())
        elif last_activation == 'step':
            self.net = nn.Sequential(*(self.net.children()),
                                     Step())            
    
    def forward(self, input_data):
        return self.net(input_data)

    def check(self, batch_size = 1):
        '''
        Print the summary of the model
        '''
        print(summary(self.net, input_size = (batch_size, self.input_size),
                      col_names = ["input_size", "output_size", "num_params"]))
        
    def _linear_block(self, input_size, output_size):
        '''
        Function for returning a block of the generator's neural network
        given input and output dimensions.
        Parameters:
            input_size: the dimension of the input vector, a scalar
            output_size: the dimension of the output vector, a scalar
        Returns:
            a linear neural network layer, with a linear transformation 
            followed by a batch normalization and then a leaky-relu activation
        '''
        if self.batch_norm:
            return nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
        else:
            return nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )

class CNN_Gen(nn.Module):

    def __init__(self, input_size):
        super(CNN_Gen, self).__init__()
        self.input_size = input_size
        
        self.l1 = nn.Sequential(nn.Linear(self.input_size, 32 * 4 ** 2))
        
        self.conv_blocks = nn.Sequential(
            # nn.BatchNorm2d(64),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            # nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 11, 3, stride=1, padding=1),
            # nn.Tanh(),
        )         
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 32, 4, 4)
        img = self.conv_blocks(out)
        return img.view(out.shape[0], -1)


class CNN_Disc(nn.Module):

    def __init__(self, input_size):
        super(CNN_Disc, self).__init__()
        self.input_size = input_size
        
        self.model = nn.Sequential(
            nn.Conv2d(11, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.25),
            # nn.BatchNorm2d(16, 0.8),
            
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.25),
            # nn.BatchNorm2d(32, 0.8),
        )
        
        self.adv_layer = nn.Sequential(nn.Linear(32 * 2 * 2, 1))
    
    def forward(self, img):
        img = img.reshape(-1, 11, 8, 8)
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        
        return validity
        
class Optuna_Gen(nn.Module):

    def __init__(self, input_size, hidden_layers_size, batch_norm=0):
        super(Optuna_Gen, self).__init__()
        self.input_size = input_size
        self.hidden_layers_size = hidden_layers_size
        self.batch_norm = batch_norm
        
        self.l1 = nn.Sequential(nn.Linear(self.input_size, hidden_layers_size[0] * 4 ** 2))

        # hidden layers
        self.net = self._cov_block(hidden_layers_size[0], hidden_layers_size[1])
        for layer in range(2, len(hidden_layers_size)):
            _hidden_layer = self._cov_block(hidden_layers_size[layer-1],
                                            hidden_layers_size[layer])
            self.net = nn.Sequential(*(self.net.children()), *(_hidden_layer.children()))

        _out_layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_layers_size[-1], 11, 3, stride=1, padding=1),
        )
        
        self.conv_blocks = nn.Sequential(*(self.net.children()), *(_out_layer.children()))
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.hidden_layers_size[0], 4, 4)
        img = self.conv_blocks(out)
        return img.view(out.shape[0], -1)
        
    def _cov_block(self, input_size, output_size):
        if self.batch_norm:
            return nn.Sequential(
                nn.Conv2d(input_size, output_size, 3, stride=1, padding=1),
                nn.BatchNorm2d(output_size, 0.8),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
        else:
            return nn.Sequential(
                nn.Conv2d(input_size, output_size, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )

class Optuna_Disc(nn.Module):

    def __init__(self, input_size, hidden_layers_size, batch_norm=0):
        super(Optuna_Disc, self).__init__()
        self.input_size = input_size
        self.batch_norm = batch_norm

        self.net = self._cov_block(11, hidden_layers_size[0])
        for layer in range(1, len(hidden_layers_size)):
            if layer < 2:
                _hidden_layer = self._cov_block(hidden_layers_size[layer-1],
                                                hidden_layers_size[layer])
            else:
                _hidden_layer = self._cov_block(hidden_layers_size[layer-1],
                                                hidden_layers_size[layer], 1)
            self.net = nn.Sequential(*(self.net.children()), *(_hidden_layer.children()))
        
        self.adv_layer = nn.Sequential(nn.Linear(hidden_layers_size[-1] * 2 * 2, 1))
    
    def forward(self, img):
        img = img.reshape(-1, 11, 8, 8)
        out = self.net(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        
        return validity
        
    def _cov_block(self, input_size, output_size, stride=2):
        if self.batch_norm:
            return nn.Sequential(
                nn.Conv2d(input_size, output_size, 3, stride=stride, padding=1),
                nn.BatchNorm2d(output_size, 0.8),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
        else:
            return nn.Sequential(
                nn.Conv2d(input_size, output_size, 3, stride=stride, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
        
if __name__ == '__main__':
    gen = DensityNN(10, [20, 30, 20], 2)
    gen.check()
import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert

# borrowed boilerplate vae code from https://github.com/AntixK/PyTorch-VAE


class VAE(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config):
        super(VAE, self).__init__(config)

        self.unsuplist = config.unsupSize




        vaelatentdim = config.vaelatentdim
        vaeinchannels = config.vaeinchannels

        modules = []
        vaehidden = [config.poolingSize]
        self.linearFC = nn.Linear(config.hidden_size, config.poolingSize)
        self.activ =  nn.ReLU()

        # Build Encoder
        self.fc_mu = nn.Linear(vaehidden[-1], vaelatentdim)
        self.fc_var = nn.Linear(vaehidden[-1], vaelatentdim)


        # Build Decoder
        modules = []

        self.decoder1 = nn.Linear(vaelatentdim, vaehidden[-1])
        self.decoder2 = nn.Linear(vaehidden[-1],int( vaehidden[-1]))

        self.logSoftmax = nn.LogSoftmax(dim=1)

        self.linearOut =  nn.ModuleList([nn.Linear   (int( vaehidden[-1]), el[0]) for el in self.unsuplist])
        self.BetaD = config.BetaD

        self.apply(self.init_bert_weights)

    def encode(self, input: torch.Tensor) :
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # result = self.activ (self.linearFC(input))

        mu = self.fc_mu(input)
        log_var = self.fc_var(input)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.activ(self.decoder1(z))
        result = self.activ(self.decoder2(result))
        outs = []


        for outputiter , linoutnetwork in enumerate(self.linearOut):
            resout = self.logSoftmax(linoutnetwork(result))
            outs.append(resout)

        outs = torch.cat((outs), dim=1)
        return outs

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, label: torch.Tensor):

        if self.BetaD==False:
            mu, log_var = self.encode(input)
            z = self.reparameterize(mu, log_var)
            return  [self.decode(z), label, mu, log_var]
        else:
            mu, log_var = self.encode(input)
            z = self.reparameterize(mu, log_var)
            return  [self.decode(z), label, mu, log_var]
    def loss_function(self,dictout) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = dictout[0].transpose(1,0)
        input = dictout[1].transpose(1,0)

        mu = dictout[2]
        log_var = dictout[3]
        if self.BetaD==False:

            kld_weight = self.config.klpar # Account for the minibatch samples from the dataset
            reconsloss = 0
            startindx = 0

            outs = []
            labs = []
            for outputiter , output in enumerate(self.unsuplist):
                elementssize = output[0]
                chunkrecons = recons[startindx:startindx+elementssize].transpose(1,0)
                labels= input[outputiter]
                lossF  = nn.NLLLoss(reduction='none', ignore_index=-1)
                temploss = lossF(chunkrecons,labels).sum()
                reconsloss =reconsloss+ temploss

                outs.append(chunkrecons)
                labs.append(labels)
                startindx = startindx+elementssize


            kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            loss = (reconsloss + kld_weight * kld_loss)/len(dictout[0])

            if self.config.klpar<1:
                self.config.klpar = self.config.klpar + 1e-5

            return {'loss': loss, 'Reconstruction_Loss':reconsloss, 'KLD':-kld_loss, 'outs':outs, 'labs':labs}
        else:


            return 0

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.vaelatentdim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]




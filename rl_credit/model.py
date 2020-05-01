from abc import abstractmethod, abstractproperty
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import rl_credit


class BaseModel:
    recurrent = False

    @abstractmethod
    def __init__(self, obs_space, action_space):
        pass

    @abstractmethod
    def forward(self, obs):
        pass


class RecurrentACModel(BaseModel):
    recurrent = True

    @abstractmethod
    def forward(self, obs, memory):
        pass

    @property
    @abstractmethod
    def memory_size(self):
        pass

##########################################################################################
# Below copied from https://github.com/lcswillems/rl-starter-files/blob/master/model.py

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


# No memory, no text
class ACModelVanilla(nn.Module, BaseModel):
    def __init__(self, obs_space, action_space):
        super().__init__()

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value


class ACModelReturnHCA(ACModelVanilla):
    def __init__(self, obs_space, action_space):
        super().__init__(obs_space, action_space)

        ## Define return-conditional HCA model
        self.return_hca = nn.Sequential(
            nn.Linear(self.image_embedding_size + 1, 64),  # +1 comes from scalar return Z
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs, z=None):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        if z is not None:
            hca_logits = self.return_hca(torch.cat((embedding, torch.unsqueeze(z,1)), 1))
            return dist, value, hca_logits

        return dist, value


class ACModelStateHCA(ACModelVanilla):
    def __init__(self, obs_space, action_space):
        super().__init__(obs_space, action_space)

        ## Define state-conditional HCA model
        self.state_hca = nn.Sequential(
            nn.Linear(self.image_embedding_size * 2, 64),  # input is 2 concated embedded images
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs, obs2=None):
        if obs.image.ndim == 3:
            # a singleton image.  Need to unsqueeze since expecting first dim
            # to correspond to batch size.
            x1 = obs.image.unsqueeze(0).transpose(1, 3).transpose(2, 3)
        else:
            x1 = obs.image.transpose(1, 3).transpose(2, 3)
        x1 = self.image_conv(x1)
        x1 = x1.reshape(x1.shape[0], -1)

        embedding1 = x1

        # evaluate policy and value function for first obs, only
        x = self.actor(embedding1)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding1)
        value = x.squeeze(1)

        if obs2 is not None:
            if obs2.image.ndim == 3:
                x2 = obs2.image.unsqueeze(0).transpose(1, 3).transpose(2, 3)
            else:
                x2 = obs2.image.transpose(1, 3).transpose(2, 3)
            x2 = self.image_conv(x2)
            x2 = x2.reshape(x2.shape[0], -1)

            embedding2 = x2

            embed_dim = embedding2.shape[1]
            if embedding1.shape[0] == 1 and embedding2.shape[0] > 1:
                # If obs contains a single image, and obs2 contains a batch of
                # images, need to replicate embedding1 (by expanding its 0th dim to the shape
                # of the num of images in obs2 to be able to concat.
                embedding1 = embedding1.squeeze()
                hca_logits = self.state_hca(
                    torch.cat((embedding1.expand(embedding2.shape[0], embed_dim),
                               embedding2), 1)
                )
            else:
                hca_logits = self.state_hca(torch.cat((embedding1, embedding2), 1))
            return dist, value, hca_logits

        return dist, value

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
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False,
                 return_embedding=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.return_embedding = return_embedding

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
        image_embedding = x

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

        if self.return_embedding:
            return dist, value, memory, image_embedding
        else:
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
    def __init__(self, obs_space, action_space, return_bins=None):
        """
        return_bins : torch.tensor
            1-dimensional, containing returns associated w/ each bin
        """
        super().__init__(obs_space, action_space)
        self._return_bins = return_bins
        if self._return_bins is not None:
            self._num_bins = len(return_bins)

        ## Define return-conditional HCA model
        if self._return_bins is not None:
            self.return_hca = nn.Sequential(
                nn.Linear(self.image_embedding_size + self._num_bins, 64),
                nn.Tanh(),
                nn.Linear(64, action_space.n)
            )
        else:
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
            if self._return_bins is not None:
                bin_idx = (torch.abs(self._return_bins.expand(z.shape[0], self._num_bins) - \
                                     z.unsqueeze(1))
                           .argmin(axis=1)
                )
                binned_return = F.one_hot(bin_idx, self._num_bins)
                hca_logits = self.return_hca(
                    torch.cat((embedding, binned_return.float()), 1))
            else:
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

        ## Define immediate reward estimated given action and obs
        self.est_reward = nn.Sequential(
            nn.Linear(self.image_embedding_size + action_space.n, 64), # action input is one hot encoded
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs, obs2=None, action=None):
        assert obs2 is None or action is None

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

            if embedding1.shape[0] == 1 and embedding2.shape[0] > 1:
                # If obs contains a single image, and obs2 contains a batch of
                # images, need to replicate embedding1 (by expanding its 0th dim to the shape
                # of the num of images in obs2 to be able to concat.
                embedding1 = (embedding1.squeeze()
                              .expand(embedding2.shape[0], embedding2.shape[1]))

            hca_logits = self.state_hca(torch.cat((embedding1, embedding2), 1))
            return dist, value, hca_logits

        if action is not None:
            if action.dim() == 1:
                reward = self.est_reward(torch.cat((embedding1, action.unsqueeze(0)), 1))
            else:
                reward = self.est_reward(torch.cat((embedding1, action), 1))
            return dist, value, reward

        return dist, value


class ACAttention(nn.Module, BaseModel):
    def __init__(self, obs_space, action_space, d_key=5):
        """
        d_key : int, dimension of image embedding projected
           to query (and key) for attention
        """
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
        n = obs_space[0]
        m = obs_space[1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define self-attention vector for input into critic
        self.d_key = d_key
        self.Wq = nn.Linear(self.image_embedding_size, d_key, bias=False)
        self.Wk = nn.Linear(self.image_embedding_size, d_key, bias=False)
        self.Wv = nn.Linear(self.image_embedding_size, d_key, bias=False)

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.d_key, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs, mask_future=True, attn_custom_mask=None):
        """
        obs : torch.tensor, dtype=float,
            of size (batch_size, ep_len, image_H, image_W, image_C)

        mask_future : bool,
            If True, apply mask so observations cannot attend to future obs, only
            previous obs.

        attn_custom_mask : torch.tensor, dtype=bool,
            mask for attention scores of dimension (batch_size, ep_len, ep_len), where
            elements that are True are to be zeroed out.

            Expected use case: mask elements outside block-diagonal so that observations
            that are in different episodes cannot attend to each other.
        """
        # expect obs to be dim 5, but for compatibility with other scripts
        # that don't have an extra dimension for sequence length, add
        # an extra dimension if obs dim is 4
        if obs.dim() == 4:
            obs = obs.unsqueeze(1)

        batch_size, ep_len, h, w, c = obs.shape

        if ep_len == 1:
            obs = torch.squeeze(obs, dim=1)
        elif ep_len > 1:
            obs = obs.reshape(batch_size * ep_len, h, w, c)

        x = obs.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        embedding = x  # dim 2: (batch_size * ep_len, embedding size)

        # policy
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        if ep_len == 1:
            # forward pass is just a single step to figure out action to take
            # in env, so skip value evaluation
            # TODO: allow value prediction for single obs / use context vector.
            # Temporary: use 0 as placeholder.
            return dist, torch.tensor([0.])

        # ==== attention before critic ====
        queries = self.Wq(embedding).view(batch_size, ep_len, self.d_key) / (self.d_key ** (1/4))
        keys = self.Wk(embedding).view(batch_size, ep_len, self.d_key) / (self.d_key ** (1/4))
        values = self.Wv(embedding).view(batch_size, ep_len, self.d_key)

        scores = torch.bmm(queries, keys.transpose(1, 2))

        # Custom mask
        if attn_custom_mask is not None:
            scores.masked_fill_(attn_custom_mask, float('-inf'))

        # Filter out self attention to future values
        if mask_future is True:
            scores_mask = torch.ones([ep_len, ep_len]).tril()
            scores.masked_fill_(scores_mask == 0, float('-inf'))

        scores = F.softmax(scores, dim=2)    # shape = (batch_size, ep_len, ep_len)
        attn_out = torch.bmm(scores, values)  # shape = (batch_size, ep_len, d_key)

        x = self.critic(attn_out.view(batch_size * ep_len, self.d_key))
        value = x.squeeze(1)

        return dist, value, scores


class AttentionQ(nn.Module, BaseModel):
    """
    Actor critic with additional Qvalue attention head
    """
    def __init__(self, obs_space, action_space, d_key=5):
        """
        d_key : int, dimension of image embedding projected
           to query (and key) for attention
        """
        super().__init__()
        self._action_space_n = action_space.n

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
        n = obs_space[0]
        m = obs_space[1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self._action_space_n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Define action embedding
        self._action_embed_size = 32
        self.action_embed = nn.Linear(self._action_space_n, self._action_embed_size)

        # Define self-attention vector for input into Qvalue head
        self.d_key = d_key
        self.Wq = nn.Linear(self.image_embedding_size + self._action_embed_size, d_key, bias=False)
        self.Wk = nn.Linear(self.image_embedding_size + self._action_embed_size, d_key, bias=False)
        self.Wv = nn.Linear(self.image_embedding_size + self._action_embed_size, d_key, bias=False)

        self.Qvalue = nn.Sequential(
            nn.Linear(self.d_key, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs, actions=None, mask_future=True, attn_custom_mask=None):
        """
        obs : torch.tensor, dtype=float,
            of size (batch_size, ep_len, image_H, image_W, image_C)

        actions : torch.tensor, dtype=float,
            of size (batch_size, ep_len, self._action_space_n)

        mask_future : bool,
            If True, apply mask so observations cannot attend to future obs, only
            previous obs.

        attn_custom_mask : torch.tensor, dtype=bool,
            mask for attention scores of dimension (batch_size, ep_len, ep_len), where
            elements that are True are to be zeroed out.

            Expected use case: mask elements outside block-diagonal so that observations
            that are in different episodes cannot attend to each other.
        """
        # expect obs to be dim 5, but for compatibility with other scripts
        # that don't have an extra dimension for sequence length, add
        # an extra dimension if obs dim is 4
        if obs.dim() == 4:
            obs = obs.unsqueeze(1)

        batch_size, ep_len, h, w, c = obs.shape

        # Reshape obs for input into actor and critic
        if ep_len == 1:
            obs = torch.squeeze(obs, dim=1)
        elif ep_len > 1:
            obs = obs.reshape(batch_size * ep_len, h, w, c)

        x = obs.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        img_embedding = x  # dim 2: (batch_size * ep_len, img embedding size)

        # policy
        x = self.actor(img_embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(img_embedding)
        value = x.squeeze(1)

        if ep_len == 1 or actions is None:
            # forward pass is just a single step to figure out action to take
            # in env, so skip Qvalue evaluation
            return dist, value

        # ===== embed one-hot encoded actions =====
        if actions is not None:
            assert actions.dim() == 3
            assert actions.shape[0] == batch_size
            assert actions.shape[1] == ep_len

        action_embedding = self.action_embed(actions.reshape(batch_size * ep_len, self._action_space_n))

        # ==== attention before Qvalue ====
        x = torch.cat((img_embedding, action_embedding), 1)

        queries = self.Wq(x).view(batch_size, ep_len, self.d_key) / (self.d_key ** (1/4))
        keys = self.Wk(x).view(batch_size, ep_len, self.d_key) / (self.d_key ** (1/4))
        values = self.Wv(x).view(batch_size, ep_len, self.d_key)

        scores = torch.bmm(queries, keys.transpose(1, 2))

        # Custom mask
        if attn_custom_mask is not None:
            scores.masked_fill_(attn_custom_mask, float('-inf'))

        # Filter out self attention to future values
        if mask_future is True:
            scores_mask = torch.ones([ep_len, ep_len]).tril()
            scores.masked_fill_(scores_mask == 0, float('-inf'))

        scores = F.softmax(scores, dim=2)    # shape = (batch_size, ep_len, ep_len)
        attn_out = torch.bmm(scores, values)  # shape = (batch_size, ep_len, d_key)

        x = self.Qvalue(attn_out.view(batch_size * ep_len, self.d_key))
        qvalue = x.squeeze(1)

        return dist, value, qvalue, scores


class QAttentionModel(nn.Module):
    """Calculate Q value given embedding input and action context vectors
    using attention.

    Creates context from concatenating embedding and action inputs.

    embedding_size : int
       size of the embedding (assumed to be a 1-D vector, not a raw RGB image)

    action_size : int
       cardinality of the action space

    d_key : int
        rank of attention matrix
    """
    def __init__(self, embedding_size, action_size, d_key=30):
        super().__init__()

        self.d_key = d_key

        # self._action_embed_size = 32
        # self.action_embed = nn.Linear(action_size, self._action_embed_size)

        # self.Wq = nn.Linear(embedding_size + self._action_embed_size, d_key, bias=False)
        # self.Wk = nn.Linear(embedding_size + self._action_embed_size, d_key, bias=False)
        # self.Wv = nn.Linear(embedding_size + self._action_embed_size, d_key, bias=False)

        self.Wq = nn.Linear(embedding_size, d_key, bias=False)
        self.Wk = nn.Linear(embedding_size, d_key, bias=False)
        self.Wv = nn.Linear(embedding_size, d_key, bias=False)

        # self.Qvalue = nn.Sequential(
        #     nn.Linear(d_key, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )
        self.Qvalue = nn.Sequential(
            nn.Linear(d_key, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, act, mask_future=True, custom_mask=None):
        """
        obs : tensor, of size
            (batch_size, sequence_length, embedding_size)

        act : tensor, of size
            (batch_size, sequence_length, action_cardinality).  Actions
            should be one-hot encoded

        mask_future: bool
            Filter out attention to future values?

        custom_mask: (optional) torch.tensor, dtype bool, of size
            (batch_size, sequence_length, sequence_length), intended for masking
            between different episodes in the same batch.
        """
        batch_sz, seq_len, _ = obs.shape

        #action_embedding = self.action_embed(act.reshape(batch_sz * seq_len, -1))

        x = obs.reshape(batch_sz * seq_len, -1)

        # batch_sz x seq_len x d_key
        queries = self.Wq(x).view(batch_sz, seq_len, self.d_key) / (self.d_key ** (1/4))
        keys = self.Wk(x).view(batch_sz, seq_len, self.d_key) / (self.d_key ** (1/4))
        values = self.Wv(x).view(batch_sz, seq_len, self.d_key)

        scores = torch.bmm(queries, keys.transpose(1, 2))

        if custom_mask is not None:
            scores.masked_fill_(custom_mask, float('-inf'))

        if mask_future is True:
            future_mask = torch.ones([seq_len, seq_len]).tril()
            scores.masked_fill_(future_mask == 0, float('-inf'))

        scores = torch.softmax(scores, dim=2)  # batch_sz x seq_len x seq_len
        attn_out = torch.bmm(scores, values)   # batch_sz x seq_len x d_key

        x = self.Qvalue(attn_out.view(batch_sz * seq_len, self.d_key))
        #qvalue = x.squeeze(1)

        #return qvalue, scores
        return x, scores

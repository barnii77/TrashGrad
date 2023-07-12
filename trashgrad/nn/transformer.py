import math
import trashgrad as tg
import numpy as np
import cupy as cp


def build_look_ahead_mask(size: int, mode="cpu"):
    lib = {"cpu": np, "gpu": cp}[mode]
    out = lib.tri(size)
    out[out == 0] = -1e9
    out[out == 1] = 0
    return out


def attention(q, k, v, mask=None):
    transpose_axes = tuple(range(len(k.shape) - 2)) + (-1, -2)
    d_k = q.shape[-1]
    if mask is not None:
        return tg.softmax(q @ k.transpose(transpose_axes) / math.sqrt(d_k) + mask) @ v
    return tg.softmax(q @ k.transpose(transpose_axes) / math.sqrt(d_k)) @ v


def positional_encoding(sequence_length, embedding_size, mode="cpu"):
    lib = {"cpu": np, "gpu": cp}[mode]
    denominator = 10000. ** (lib.arange(0, embedding_size, 2) / embedding_size)
    position = lib.arange(sequence_length).reshape((-1, 1))
    stacked = lib.stack([lib.sin(position / denominator), lib.cos(position / denominator)], axis=2)
    PE = stacked.reshape((sequence_length, -1))
    return tg.tensor(PE, mode=mode)


def build_padding_mask(sequence_lengths, mode="cpu"):
    """sequence_lengths: numpy.ndarray | cupy.ndarray based on if you are using gpu or cpu"""
    assert sequence_lengths.ndim == 1
    lib = {"cpu": np, "gpu": cp}[mode]
    array_module = cp.get_array_module(sequence_lengths)
    if array_module is not lib:
        if lib is np:
            sequence_lengths = cp.asnumpy(sequence_lengths)
        else:
            sequence_lengths = cp.asarray(sequence_lengths)
    max_seq_len = lib.max(sequence_lengths)
    indices = lib.tile(lib.arange(max_seq_len), (sequence_lengths.size, 1))
    mask = lib.zeros(indices.shape)
    mask[indices > sequence_lengths.reshape((-1, 1))] = -1e9
    return tg.tensor(mask, mode=mode)


class DiscreteEmbedding(tg.Module):
    """A special case of the Embedding class where the input feature tokens of the forward method would be one-hot encoded if one were using the standard Embedding class.
    In this case, one has to pass in the hot positions (the ones) directly as indices instead of one-hot encoding them."""

    def __init__(self, vocab_size: int, d_model: int, mode="cpu"):
        self.embeddings = tg.tensor(np.random.randn((vocab_size, d_model)), requires_grad=True, mode=mode)
        self.features = [self.embeddings]

    def forward(self, tokens):
        return self.embeddings[tokens]


class Embedding(tg.Module):
    def __init__(self, vocab_size: int, d_model: int, mode="cpu"):
        self.embeddings = tg.Dense(vocab_size, d_model, mode=mode)
        self.features = [self.embeddings]

    def forward(self, vec: tg.Tensor):
        return self.embeddings(vec)


class MultiHeadAttention(tg.Module):
    def __init__(self, d_model: int, num_heads: int = 8):
        self.qkv_layer = tg.Dense(d_model, 3 * d_model)
        self.final_layer = tg.Dense(d_model, d_model)
        self.num_heads = num_heads
        self.d_model = d_model

    def forward(self, x: tg.Tensor, mask: tg.Tensor = None) -> tg.Tensor:
        assert len(x.shape) == 3
        batch_size, sequence_length, d_model = x.shape
        return self.final_layer(attention(*self.qkv_layer(x)
                                          .reshape((batch_size, sequence_length, self.num_heads, -1))
                                          .transpose((0, 2, 1, 3))
                                          .chunk(3), mask=mask)
                                .transpose((0, 2, 1, 3))
                                .reshape((batch_size, sequence_length, d_model)))


class MultiHeadCrossAttention(tg.Module):
    def __init__(self, d_model: int, num_heads: int = 8):
        self.q_layer = tg.Dense(d_model, d_model)
        self.kv_layer = tg.Dense(d_model, 2 * d_model)
        self.final_layer = tg.Dense(d_model, d_model)
        self.num_heads = num_heads
        self.d_model = d_model

    def forward(self, x: tg.Tensor, y: tg.Tensor, mask: tg.Tensor = None) -> tg.Tensor:
        batch_size, prompt_sequence_length, d_model = x.shape
        _, completion_sequence_length, _ = y.shape
        q = (self.q_layer(y)
             .reshape((batch_size, completion_sequence_length, self.num_heads, -1))
             .transpose((0, 2, 1, 3)))
        k, v = (self.kv_layer(x)
                .reshape((batch_size, prompt_sequence_length, self.num_heads, -1))
                .transpose((0, 2, 1, 3))
                .chunk(2))
        return self.final_layer(attention(q, k, v, mask=mask)
                                .transpose((0, 2, 1, 3))
                                .reshape((batch_size, completion_sequence_length, d_model)))


class PostLNEncoderLayer(tg.Module):
    def __init__(self, feed_forward: tg.Module, d_model: int, num_heads: int = 8, drop_prob: float = .1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = feed_forward
        self.LN1 = tg.LayerNorm(d_model)  # todo: add a mode to transformer module
        self.LN2 = tg.LayerNorm(d_model)

    def forward(self, prompt: tg.Tensor, padding_mask: tg.Tensor = None):
        if padding_mask is not None:
            if prompt.ndim > padding_mask.ndim:
                padding_mask = padding_mask.reshape(
                    (padding_mask.shape[0],) + tuple(1 for _ in range(prompt.ndim - padding_mask.ndim)) + (
                        padding_mask.shape[1],))
        residual = self.LN1(
            prompt + tg.dropout(self.multi_head_attention(prompt, mask=padding_mask), self.drop_prob)
        )
        return self.LN2(residual + tg.dropout(self.feed_forward(residual), self.drop_prob))


class PostLNEncoder(tg.Module):
    def __init__(self, feed_forward_architecture, num_encoder_layers: int, d_model: int, num_attention_heads: int = 8,
                 drop_prob: float = .1):
        self.model = tg.Sequential(
            [PostLNEncoderLayer(feed_forward_architecture(), d_model, num_attention_heads, drop_prob) for _ in
             range(num_encoder_layers)])

    def forward(self, x: tg.Tensor):
        assert len(x.shape) == 3
        return self.model(x)


class PostLNDecoderLayer(tg.Module):
    def __init__(self, feed_forward: tg.Module, d_model: int, num_heads: int = 8, drop_prob: float = .1):
        self.feed_forward = feed_forward
        self.d_model = d_model
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.masked_multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.multi_head_cross_attention = MultiHeadCrossAttention(d_model, num_heads)

    def forward(self, completion: tg.Tensor, prompt: tg.Tensor, look_ahead_mask: tg.Tensor,
                completion_padding_mask: tg.Tensor = None, prompt_padding_mask: tg.Tensor = None):
        if prompt_padding_mask is not None:
            if prompt.ndim > prompt_padding_mask.ndim:
                prompt_padding_mask = prompt_padding_mask.reshape((prompt_padding_mask.shape[0],) + tuple(
                    1 for _ in range(prompt.ndim - prompt_padding_mask.ndim)) + (prompt_padding_mask.shape[1],))
        if completion_padding_mask is not None:
            if completion.ndim > completion_padding_mask.ndim:
                completion_padding_mask = completion_padding_mask.reshape((completion_padding_mask.shape[0],) + tuple(
                    1 for _ in range(completion.ndim - completion_padding_mask.ndim)) + (
                                                                              completion_padding_mask.shape[1],))
            residual = tg.standardize(completion + tg.dropout(
                self.masked_multi_head_attention(completion, mask=look_ahead_mask + completion_padding_mask),
                self.drop_prob))
        else:
            residual = tg.standardize(
                completion + tg.dropout(self.masked_multi_head_attention(completion, mask=look_ahead_mask),
                                        self.drop_prob))
        residual = tg.standardize(
            residual + tg.dropout(self.multi_head_cross_attention(prompt, completion, mask=prompt_padding_mask),
                                  self.drop_prob))
        return tg.standardize(residual + tg.dropout(self.feed_forward(residual), self.drop_prob))


class PostLNDecoder(tg.Module):
    def __init__(self, feed_forward_architecture, num_encoder_layers: int, d_model: int, num_attention_heads: int = 8,
                 drop_prob: float = .1):
        self.layers = [PostLNDecoderLayer(feed_forward_architecture(), d_model, num_attention_heads, drop_prob) for _ in
                       range(num_encoder_layers)]

    def forward(self, completion: tg.Tensor, prompt: tg.Tensor, look_ahead_mask: tg.Tensor,
                completion_padding_mask: tg.Tensor, prompt_padding_mask: tg.Tensor):
        assert len(completion.shape) == 3
        out = completion
        for layer in self.layers:
            out = layer(out, prompt, look_ahead_mask, completion_padding_mask, prompt_padding_mask)
        return out


class ReluFeedForwardUnit(tg.Module):
    def __init__(self, hidden_size: int, inout_size: int):
        self.model = tg.Sequential([tg.Dense(inout_size, hidden_size), tg.relu, tg.Dense(hidden_size, inout_size)])

    def forward(self, x: tg.Tensor):
        return self.model(x)


class PostLNTransformer(tg.Module):
    def __init__(self, prompt_embedding: Embedding, completion_embedding: Embedding, final_classifier: tg.Module, num_encoder_layers: int, num_decoder_layers: int,
                 feed_forward_architecture=ReluFeedForwardUnit, d_model: int = 512, num_attention_heads: int = 8,
                 drop_prob: float = .1):
        self.prompt_embedding = prompt_embedding
        self.completion_embedding = completion_embedding
        self.encoder = PostLNEncoder(feed_forward_architecture, num_encoder_layers, d_model, num_attention_heads,
                                     drop_prob)
        self.decoder = PostLNDecoder(feed_forward_architecture, num_decoder_layers, d_model, num_attention_heads,
                                     drop_prob)
        self.final_classifier = final_classifier

    def forward(self, prompt: tg.Tensor, completion: tg.Tensor, look_ahead_mask: tg.Tensor, completion_padding_mask: tg.Tensor, prompt_padding_mask: tg.Tensor):
        """prompt and completion are 2d tensors of indices of tokens, like [[3, 2, 99], [4, 2, 1]]. 99 here would be padding"""
        prompt_embedding = self.prompt_embedding(prompt)
        completion_embedding = self.completion_embedding(completion)
        encoder_out = self.encoder(prompt_embedding)
        decoder_out = self.decoder(completion_embedding, encoder_out, look_ahead_mask, completion_padding_mask, prompt_padding_mask)
        return tg.softmax(self.final_classifier(decoder_out))


# todo: look_ahead_mask should be working, padding_mask needs a reshape to shape (batch_size, 1, 1, ..., 1, sequence_length) and its base-form
# todo: of shape (batch_size, sequence_length) should look like this:

# [-inf -inf -inf -inf]
# [-inf    0    0 -inf]
# [   0    0    0    0]
# [   0    0    0    0]

# todo: where 0 is no padding (not masked) and -inf is masked. this will then be broadcasted by numpy/cupy appropriately.

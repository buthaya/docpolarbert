import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss

from transformers.activations  import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    TokenClassifierOutput,
)

from transformers.modeling_utils import PreTrainedModel
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMOnlyMLMHead, MaskedLMOutput
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import (
    logging,
)
from .configuration_docpolarbert import DocPolarBERTConfig


logger = logging.get_logger(__name__)

class DocPolarBERTTextEmbeddings(nn.Module):
    """
    DocPolarBERT text embeddings. Same as `RobertaEmbeddings` but with added spatial (layout) embeddings.
    """

    def __init__(self, config):
        # WORD EMBEDDINGS + 1D POSITION EMBEDDINGS
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=0  # replaced self.padding_idx with 0
        )

    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:

        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
        return incremental_indices.long()  # removed + padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded. Replaced padding_idx with 0
                position_ids = self.create_position_ids_from_input_ids(input_ids=input_ids, padding_idx=0).to(
                    input_ids.device
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class DocPolarBERTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DocPolarBERTConfig
    base_model_prefix = "docpolarbert"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class DocPolarBERTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias
        self.has_angle_attention_bias = config.has_angle_attention_bias

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def cogview_attention(self, attention_scores, alpha=32):
        """
        https://arxiv.org/abs/2105.13290 Section 2.4 Stabilization of training: Precision Bottleneck Relaxation
        (PB-Relax). A replacement of the original nn.Softmax(dim=-1)(attention_scores). Seems the new attention_probs
        will result in a slower speed and a little bias. Can use torch.allclose(standard_attention_probs,
        cogview_attention_probs, atol=1e-08) for comparison. The smaller atol (e.g., 1e-08), the better.
        """
        scaled_attention_scores = attention_scores / alpha
        max_value = scaled_attention_scores.amax(dim=(-1)).unsqueeze(-1)
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        return nn.Softmax(dim=-1)(new_attention_scores)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
        rel_2d_angle=None,
    ):
        """
        # B = Batch size, S = Sequence Length, N = Number of attention heads, H = Size of each head
        # S_q = Sequence length of query, S_k = Sequence length of key, S_v = Sequence length of value

        :param hidden_states: (B, S, H)
        :param attention_mask:
        :param head_mask:
        :param output_attentions: bool
        :param rel_pos:  (B, S_q, S_k, H)
        :param rel_2d_pos: (B, S_q, S_k, H)
        :param rel_2d_angle: (B, S_q, S_k, H)
        :return:
        """

        mixed_query_layer = self.query(hidden_states) # (B, S_q, N*H)

        key_layer = self.transpose_for_scores(self.key(hidden_states)) # (B, N, S_q, H)
        value_layer = self.transpose_for_scores(self.value(hidden_states)) # (B, N, S_k, H)
        query_layer = self.transpose_for_scores(mixed_query_layer) # (B, N, S_v, H)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # The attention scores QT K/√d could be significantly larger than input elements, and result in overflow.
        # Changing the computational order into QT(K/√d) alleviates the problem. (https://arxiv.org/pdf/2105.13290.pdf)
        attention_scores_1 = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        # intialize zeros relative attention_scores
        attention_scores_rel_1D = 0
        attention_scores_rel_2D = 0
        attention_scores_rel_angle = 0

        if self.has_relative_attention_bias:
            # Compute the 1D-relative position attention scores using einsum for efficient batching
            # 'bnsh, nskh -> bnsk' indicates:
            # b: batch, n: head, s: sequence positions, h: head dimension k: sequence positions for keys
            # Resulting attn: (B, N, S_q, S_k)
            attention_scores_rel_1D = torch.einsum('bnsh, bskh -> bnsk', query_layer / math.sqrt(self.attention_head_size),
                                              rel_pos)

        if self.has_spatial_attention_bias:
            # Compute the 2D-relative position attention scores using einsum for efficient batching
            # 'bnsh, bskh -> bnsk' indicates:
            # b: batch, n: head, s: sequence positions, h: head dimension k: sequence positions for keys
            # Resulting attn: (B, N, S_q, S_k)
            attention_scores_rel_2D = torch.einsum('bnsh, bskh -> bnsk', query_layer / math.sqrt(self.attention_head_size),
                                              rel_2d_pos)

        if self.has_angle_attention_bias:
            # Compute the 2D-relative angle attention scores using einsum for efficient batching
            # 'bnsh, bskh -> bnsk' indicates:
            # b: batch, n: head, s: sequence positions, h: head dimension k: sequence positions for keys
            # Resulting attn: (B, N, S_q, S_k)
            attention_scores_rel_angle = torch.einsum('bnsh, bskh -> bnsk', query_layer / math.sqrt(self.attention_head_size),
                                              rel_2d_angle)
        # Sum the 1D, 2D and angle relative position scores to the query-key attention scores
        attention_scores = attention_scores_1 + attention_scores_rel_1D + attention_scores_rel_2D + attention_scores_rel_angle

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # Use the trick of the CogView paper to stablize training
        attention_probs = self.cogview_attention(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfOutput
class DocPolarBERTSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.layoutlmv2.modeling_layoutlmv2.LayoutLMv2Attention with LayoutLMv2->DocPolarBERT
class DocPolarBERTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = DocPolarBERTSelfAttention(config)
        self.output = DocPolarBERTSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
        rel_2d_angle=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
            rel_2d_angle=rel_2d_angle,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.layoutlmv2.modeling_layoutlmv2.LayoutLMv2Layer with LayoutLMv2->DocPolarBERT
class DocPolarBERTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = DocPolarBERTAttention(config)
        self.intermediate = DocPolarBERTIntermediate(config)
        self.output = DocPolarBERTOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
        rel_2d_angle=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
            rel_2d_angle=rel_2d_angle,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class DocPolarBERTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([DocPolarBERTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_bias = nn.Linear(self.rel_pos_bins, self.attention_head_size, bias=False)

        if self.has_spatial_attention_bias:
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_angle_bins = config.rel_2d_angle_bins
            # self.rel_2d_pos_bins + 2 for 0 and padding_idx
            self.rel_pos_2D_embedding = nn.Embedding(self.rel_2d_pos_bins +2, self.attention_head_size, padding_idx=self.rel_2d_pos_bins+1)
            # self.rel_2d_angle_bins + 2 for 0
            # padding_idx = self.rel_2d_angle_bins+4+1 because we have 4 special cases for angles close to the axes
            self.rel_pos_angle_embedding = nn.Embedding(self.rel_2d_angle_bins + 4 + 2, self.attention_head_size, padding_idx=self.rel_2d_angle_bins+4+1)

    def relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        ret = 0
        if bidirectional:
            num_buckets //= 2
            ret += (relative_position > 0).long() * num_buckets
            n = torch.abs(relative_position)
        else:
            n = torch.max(-relative_position, torch.zeros_like(relative_position))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def _cal_1d_pos_emb(self, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)

        rel_pos = self.relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        # Since this is a simple indexing operation that is independent of the input,
        # no need to track gradients for this operation
        #
        # Without this no_grad context, training speed slows down significantly
        with torch.no_grad():
            rel_pos = self.rel_pos_bias.weight.t()[rel_pos]
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def group_distances(self, dist, num_buckets, padding_value):
        """
        Bins each row of the eucl_dist matrix into B equal-sized groups based on quantiles.
        Group numbers range from 1 to B, and diagonal elements are set to 0.

        :param dist: A 3D tensor (batch_size, num_rows, num_columns) representing distances.
        :param num_buckets: Number of groups to divide the distances into.
        :param padding_value: Value to be treated as padding and excluded from quantile calculation.

        :return: A tensor with the same shape as dist, where the distances are replaced by
                 group numbers (from 1 to num_buckets), and diagonal elements are 0.
        """
        # Mask out the padding values for quantile calculation
        mask = dist.ne(padding_value)
        # Extract valid (non-padding) distances
        valid_distances = dist.masked_select(mask)
        if len(valid_distances) == 0:
            return torch.full_like(dist, padding_value).to(torch.long)  # Return all padding values if no valid entries

        # Compute quantiles for valid distances across the entire tensor
        quantiles = torch.quantile(valid_distances, torch.linspace(0, 1, num_buckets + 1, device=dist.device))

        # Create the output tensor with the same shape as the input
        distances = torch.full_like(dist, padding_value, dtype=torch.float32)
        # Assign group numbers based on quantile bins
        bucketized = torch.bucketize(dist, quantiles, right=False).to(torch.float32)

        # Replace values in the distances tensor where mask is True
        distances[mask] = bucketized[mask]

        return distances.to(torch.long)


    def group_angles(self, bbox, angles, num_quadrants, padding_value):
        """
        Discretizes a matrix of angles into Q general quadrants, adding special cases for angles
        close to the axes (x-axis and y-axis), and checks for intersections on x-axis or y-axis
        based on box coordinates.
        Special cases:
        - Close to 0° or 360° (x-axis, left to right direction).
        - Close to 90° (y-axis, bottom to top direction).
        - Close to 180° (x-axis, right to left direction).
        - Close to 270° (y-axis, top to bottom direction).
        Additional conditions:
        - If two boxes intersect on the x-axis, put them in quadrant close_to_90 or close_to_270.
        - If two boxes intersect on the y-axis, put them in quadrant close_to_0_or_360 or close_to_180.

        :param angle_matrix: BxNxN matrix where element [i, j] is the angle between box i and j.
        :param boxes: BxNx4 array where each row is [x1, y1, x2, y2] representing the coordinates of box i.
        :param num_quadrants: Number of general quadrants.
        :param padding_value: Value to use for padding elements.

        :return: NxN matrix where element [i, j] is the quadrant number (1 to num_quadrants+special cases),
                 and diagonal elements are 0.
        """
        binned_matrix = torch.zeros_like(angles, device=angles.device)

        general_quadrant_size = 360 / num_quadrants

        box_i_min_x = bbox[:, :, 0].unsqueeze(2)
        box_i_max_x = bbox[:, :, 2].unsqueeze(2)
        box_j_min_x = bbox[:, :, 0].unsqueeze(1)
        box_j_max_x = bbox[:, :, 2].unsqueeze(1)

        box_i_min_y = bbox[:, :, 1].unsqueeze(2)
        box_i_max_y = bbox[:, :, 3].unsqueeze(2)
        box_j_min_y = bbox[:, :, 1].unsqueeze(1)
        box_j_max_y = bbox[:, :, 3].unsqueeze(1)

        # Intersection on x-axis and y-axis
        intersect_x = (box_i_max_x >= box_j_min_x) & (box_j_max_x >= box_i_min_x)
        intersect_y = (box_i_max_y >= box_j_min_y) & (box_j_max_y >= box_i_min_y)

        angles_mod = angles % 360

        mask_padding = (angles_mod == padding_value)
        binned_matrix[mask_padding] = padding_value

        # Special cases
        mask_case_1 = (intersect_y & ((angles_mod < 90) | (angles_mod > 270)))  # Close to 0° or 360°
        mask_case_2 = (intersect_y & (angles_mod > 90) & (angles_mod < 270))  # Close to 180°
        mask_case_3 = (intersect_x & (angles_mod < 180))  # Close to 90°
        mask_case_4 = (intersect_x & (angles_mod > 180))  # Close to 270°
        binned_matrix[mask_case_1] = 1
        binned_matrix[mask_case_2] = 3
        binned_matrix[mask_case_3] = 2
        binned_matrix[mask_case_4] = 4

        # Other cases
        general_quadrant = (angles_mod // general_quadrant_size) + 1
        binned_matrix[~mask_padding & ~(mask_case_1 | mask_case_2 | mask_case_3 | mask_case_4)] = general_quadrant[~mask_padding & ~(mask_case_1 | mask_case_2 | mask_case_3 | mask_case_4)] + 4
        diagonal_mask = torch.eye(angles.size(1), device=angles.device).unsqueeze(0).repeat(angles.size(0), 1, 1).bool()
        binned_matrix[diagonal_mask] = 0

        return binned_matrix.to(torch.long)

    def _cal_2D_pos_emb(self, bbox):
        x1 = bbox.clone().detach()[:, :, 0]
        y1 = bbox.clone().detach()[:, :, 3]
        x2 = bbox.clone().detach()[:, :, 2]
        y2 = bbox.clone().detach()[:, :, 1]

        centers = (bbox[..., :2] + bbox[..., 2:]) / 2
        diffs = centers[:, :, None, :] - centers[:, None, :, :]
        squared_diffs = diffs.pow(2).sum(-1)
        dist = torch.sqrt(squared_diffs)
        # get indices of the boxes that are [0, 0, 0, 0] or [1000, 1000, 1000, 1000]
        mask = (x1 == x2) & (y1 == y2)
        # Get the mask in [512, 512] shape
        mask = mask.unsqueeze(1) | mask.unsqueeze(2)
        # Set the distances between padding boxes to 0
        padding_value = self.rel_2d_pos_bins+1
        dist[mask] = padding_value

        rel_pos_2D = self.group_distances(dist, num_buckets=self.rel_2d_pos_bins, padding_value=padding_value)
        rel_pos_2D = self.rel_pos_2D_embedding(rel_pos_2D)
        rel_pos_2D = rel_pos_2D.to(torch.float32)
        rel_pos_2D = rel_pos_2D.contiguous()
        return rel_pos_2D

    def _cal_2D_angle_emb(self, bbox):
        x1 = bbox.clone().detach()[:, :, 0]
        y1 = bbox.clone().detach()[:, :, 3]
        x2 = bbox.clone().detach()[:, :, 2]
        y2 = bbox.clone().detach()[:, :, 1]

        centers = (bbox[..., :2] + bbox[..., 2:]) / 2

        # Calculate the angle between each pair of boxes
        angles = torch.zeros(centers.size(0), centers.size(1), centers.size(1), device=bbox.device)

        angles[:, :, :] = (360 - torch.atan2(centers[:, :, 1].unsqueeze(1) - centers[:, :, 1].unsqueeze(2),
                                             centers[:, :, 0].unsqueeze(1) - centers[:, :, 0].unsqueeze(
                                                 2)) * 180 / torch.pi) % 360

        # get indices of the boxes that are [0, 0, 0, 0] or [1000, 1000, 1000, 1000]
        mask = (x1 == x2) & (y1 == y2)
        # Get the mask in [512, 512] shape
        mask = mask.unsqueeze(1) | mask.unsqueeze(2)
        # Set the distances between padding boxes to 0
        padding_value = self.rel_2d_angle_bins+5
        angles[mask] = padding_value
        rel_2d_angle = self.group_angles(bbox, angles, num_quadrants=self.rel_2d_angle_bins, padding_value=padding_value)
        rel_2d_angle = self.rel_pos_angle_embedding(rel_2d_angle)
        rel_2d_angle = rel_2d_angle.to(torch.float32)
        rel_2d_angle = rel_2d_angle.contiguous()
        return rel_2d_angle

    def forward(
        self,
        hidden_states,
        bbox=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        position_ids=None,
        patch_height=None,
        patch_width=None,
    ):
        device = hidden_states.device
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        rel_pos = self._cal_1d_pos_emb(position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2D_pos_emb(bbox) if self.has_spatial_attention_bias else None
        rel_2d_angle = self._cal_2D_angle_emb(bbox) if self.has_spatial_attention_bias else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    rel_pos,
                    rel_2d_pos,
                    rel_2d_angle
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                    rel_2d_angle = rel_2d_angle
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Copied from transformers.models.roberta.modeling_roberta.RobertaIntermediate
class DocPolarBERTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.roberta.modeling_roberta.RobertaOutput
class DocPolarBERTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DocPolarBERTModel(DocPolarBERTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.text_embed:
            self.embeddings = DocPolarBERTTextEmbeddings(config)

        self.encoder = DocPolarBERTEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def init_visual_bbox(self, image_size=(14, 14), max_len=1000):
        """
        Create the bounding boxes for the visual (patch) tokens.
        """
        visual_bbox_x = torch.div(
            torch.arange(0, max_len * (image_size[1] + 1), max_len), image_size[1], rounding_mode="trunc"
        )
        visual_bbox_y = torch.div(
            torch.arange(0, max_len * (image_size[0] + 1), max_len), image_size[0], rounding_mode="trunc"
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_size[0], 1),
                visual_bbox_y[:-1].repeat(image_size[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(image_size[0], 1),
                visual_bbox_y[1:].repeat(image_size[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, 4)

        cls_token_box = torch.tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])
        self.visual_bbox = torch.cat([cls_token_box, visual_bbox], dim=0)

    def calculate_visual_bbox(self, device, dtype, batch_size):
        visual_bbox = self.visual_bbox.repeat(batch_size, 1, 1)
        visual_bbox = visual_bbox.to(device).type(dtype)
        return visual_bbox

    def forward_image(self, pixel_values):
        embeddings = self.patch_embed(pixel_values)

        # add [CLS] token
        batch_size, seq_len, _ = embeddings.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add position embeddings
        if self.pos_embed is not None:
            embeddings = embeddings + self.pos_embed

        embeddings = self.pos_drop(embeddings)
        embeddings = self.norm(embeddings)

        return embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device

        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            if bbox is None:
                bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )

        final_bbox = final_position_ids = None
        patch_height = patch_width = None

        if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
            if self.config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, : input_shape[1]]
                position_ids = position_ids.expand_as(input_ids)
                final_position_ids = position_ids

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, None, device, dtype=embedding_output.dtype
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            patch_height=patch_height,
            patch_width=patch_width,
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class DocPolarBERTClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks. Reference: RobertaClassificationHead
    """

    def __init__(self, config, pool_feature=False):
        super().__init__()
        self.pool_feature = pool_feature
        if pool_feature:
            self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class PredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_positions = 11 # Number of max positions to predict (from 0 to 10)
        self.position_predictor = nn.Linear(config.hidden_size, self.num_positions)

    def forward(
            self,
            hidden_states,
            masked_position_ids,
            labels_pos_ids):
        # Compute the position logits (predicted positions)
        position_logits = self.position_predictor(hidden_states)  # (batch_size, sequence_length, num_positions)
        # Create mask for positions to predict (masked positions where masked_position_ids == -1)
        masked_positions_mask = masked_position_ids == -1  # Boolean mask of shape (batch_size, sequence_length)
        # Flatten the mask for easy indexing
        masked_positions_mask_flat = masked_positions_mask.view(-1)
        # Extract the logits for masked positions
        # Flatten the position logits to make them easier to work with (batch_size * sequence_length, num_positions)
        position_logits_flat = position_logits.view(-1, position_logits.size(-1))
        # Mask the logits to only consider valid predictions for masked positions
        masked_logits = position_logits_flat[masked_positions_mask_flat]
        # Get the corresponding labels for masked positions (where labels_pos_ids != -100)
        masked_labels = labels_pos_ids[masked_positions_mask]
        # Compute the loss for position prediction using cross-entropy
        if masked_labels.numel() > 0:  # Check if there are any masked positions to predict
            loss = F.cross_entropy(masked_logits, masked_labels)
        else:
            loss = torch.tensor(0.0, device=hidden_states.device)

        return loss, position_logits


class DocPolarBERTForMaskedLM(DocPolarBERTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.docpolarbert = DocPolarBERTModel(config)
        self.cls = LayoutLMOnlyMLMHead(config)
        self.position_predictor = PredictionHead(config)
        self.pos_coeff = 0.5

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        masked_position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_pos_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.docpolarbert(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # pixel_values=pixel_values,
        )

        sequence_output = outputs[0]

        # MLM scores
        prediction_scores = self.cls(sequence_output)
        # MLM loss
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
        # Position prediction loss
        loss_pos = None
        if labels_pos_ids is not None:
            loss_pos, _ = self.position_predictor(sequence_output, masked_position_ids, labels_pos_ids)
        final_loss = masked_lm_loss+self.pos_coeff*loss_pos
        if final_loss is None:
            print('masked_lm_loss:', masked_lm_loss)
            print('prediction_scores:', prediction_scores)
            print('labels:', labels)
            print('loss_pos:', loss_pos)
            print('labels_pos_ids:', labels_pos_ids)
            print('masked_position_ids:', masked_position_ids)
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((final_loss,) + output) if final_loss is not None else output
            )

        return MaskedLMOutput(
            loss=final_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DocPolarBERTForTokenClassification(DocPolarBERTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.docpolarbert = DocPolarBERTModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.num_labels < 10:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = DocPolarBERTClassificationHead(config, pool_feature=False)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.docpolarbert(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # pixel_values=pixel_values,
        )
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


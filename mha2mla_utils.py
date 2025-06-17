import torch

def partial_rope_mask(model_args, mha2mla_args):
    """
    Generate different types of masks for partial rotary position embeddings (RoPE)
    based on configuration settings.
    Returns:
        Appropriate mask tensor based on the specified version
    """
    n_head = model_args.num_attention_heads
    n_k_head = model_args.num_key_value_heads
    if hasattr(model_args, "head_dim"):
        d_head = model_args.head_dim
    else:
        d_head = model_args.hidden_size // n_head
    d_head_half = d_head // 2
    rope_dim_for_mla = mha2mla_args.rope_dim_for_mla
    rope_dim_for_mla_half = rope_dim_for_mla // 2
    rope_version = mha2mla_args.partial_rope_version
    mask = torch.zeros(d_head)

    def select_high_frequency(mask):
        """
        Select high-frequency components (first rope_dim_for_mla dimensions)
        Returns:
            mask: Binary mask with 1s for the first rope_dim_for_mla dimensions
        """
        mask[:rope_dim_for_mla_half] = 1
        mask[d_head_half : d_head_half + rope_dim_for_mla_half] = 1
        q_masks = mask.repeat(n_head).bool()
        k_masks = mask.repeat(n_k_head).bool()
        return q_masks, k_masks

    def select_low_frequency(mask):
        """
        Select low-frequency components (last rope_dim_for_mla dimensions)
        Returns:
            mask: Binary mask with 1s for the last rope_dim_for_mla dimensions
        """
        mask[d_head - rope_dim_for_mla_half :] = 1
        mask[d_head_half - rope_dim_for_mla_half : d_head_half] = 1
        q_masks = mask.repeat(n_head).bool()
        k_masks = mask.repeat(n_k_head).bool()
        return q_masks, k_masks

    def select_uniform_frequency(mask, start_point):
        """
        Select uniformly distributed dimensions for RoPE
        Returns:
            mask: Binary mask with 1s at uniformly spaced positions
        """
        step = d_head // rope_dim_for_mla
        assert d_head_half % step == 0, "rope_dim_for_mla must be greater than 0"

        for i in range(start_point, d_head, step):
            mask[i] = 1
        q_masks = mask.repeat(n_head).bool()
        k_masks = mask.repeat(n_k_head).bool()
        return q_masks, k_masks

    def select_2norm_frequency(mask, rope_dim_for_mla):
        """
        Select dimensions based on 2-norm frequency importance
        Returns:
            mask: Binary mask with 1s for the top rope_dim_for_mla dimensions by 2-norm
        """
        # This is a placeholder implementation since the exact 2-norm selection
        # method was not detailed in the comments. In practice, this would
        # require statistics from the weight matrices to determine importance.
        with open(mha2mla_args.qk_tensor_path, "rb") as fin:
            qk_norm_rank = torch.load(fin, weights_only=True)

        k_masks = qk_norm_rank < rope_dim_for_mla_half
        if mha2mla_args.is_gqa2mha2mla:
            q_masks = k_masks
        else:
            q_masks = k_masks.repeat_interleave(n_head // n_k_head, dim=1)
        k_masks = k_masks.view(k_masks.size(0), -1)
        q_masks = q_masks.view(q_masks.size(0), -1)
        return q_masks, k_masks

    if rope_version == "high":
        return select_high_frequency(mask)
    elif rope_version == "low":
        return select_low_frequency(mask)
    elif rope_version == "uniform":
        return select_uniform_frequency(mask, mha2mla_args.uniform_start_point)
    elif rope_version == "2-norm":
        return select_2norm_frequency(d_head, rope_dim_for_mla)


def reorder_matrix_rows(mask, is_cat=False):
    """
    Reorder rows in a matrix based on a binary mask.
    Rows corresponding to 1s in the mask come first, then rows corresponding to 0s.

    Args:
        weight: The weight matrix to reorder
        mask: A binary mask (list or tensor) of length equal to weight.shape[0]

    Returns:
        The reordered weight matrix
    """
    ones_indices = torch.where(mask)[0]
    zeros_indices = torch.where(~mask)[0]
    if is_cat:
        return torch.cat([ones_indices, zeros_indices])
    else:
        return ones_indices, zeros_indices
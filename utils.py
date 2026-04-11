import torch


def histogram_matching(
    inp: torch.Tensor,
    target: torch.Tensor,
    blocksize: list = [4, 4],
):
    """
    Performs histogram matching from inp (PAN) to target (MS).
    Can operate globally or block-wise.
    inp: (B, 1, H, W)
    target: (B, C, h, w)
    Output: (B, C, H, W)
    """
    C_target = target.shape[1]

    def block_mapping(inp_block, target_block):
        """
        Performs histogram matching for a batch of blocks.
        inp_block: (N, 1, H_pan_block, W_pan_block)
        target_block: (N, C_target, H_ms_block, W_ms_block)
        Output: (N, C_target, H_pan_block, W_pan_block)
        """
        mean_I = torch.mean(inp_block, dim=(-1, -2), keepdim=True)  # (N, 1, 1, 1)
        std_I = torch.std(inp_block, dim=(-1, -2), keepdim=True)  # (N, 1, 1, 1)

        mean_T_channelwise = torch.mean(
            target_block, dim=(-1, -2), keepdim=True
        )  # (N, C_target, 1, 1)
        std_T_channelwise = torch.std(
            target_block, dim=(-1, -2), keepdim=True
        )  # (N, C_target, 1, 1)

        matched = (inp_block - mean_I) * (
            std_T_channelwise / (std_I + 1e-6)
        ) + mean_T_channelwise
        return matched

    if not blocksize[0]:
        mean_I_global = torch.mean(inp, dim=(-1, -2), keepdim=True)  # (B, 1, 1, 1)
        std_I_global = torch.std(inp, dim=(-1, -2), keepdim=True)  # (B, 1, 1, 1)

        mean_T_global_channelwise = torch.mean(
            target, dim=(-1, -2), keepdim=True
        )  # (B, C_target, 1, 1)
        std_T_global_channelwise = torch.std(
            target, dim=(-1, -2), keepdim=True
        )  # (B, C_target, 1, 1)

        pan_match = (inp - mean_I_global) * (
            std_T_global_channelwise / (std_I_global + 1e-6)
        ) + mean_T_global_channelwise

    else:
        ratio = int(inp.shape[-1] / target.shape[-1])
        h_ms, w_ms = target.shape[-2:]
        ph, pw = blocksize[0], blocksize[1]

        num_h_blocks = h_ms // ph
        num_w_blocks = w_ms // pw

        inp_blocks = (
            inp.unfold(2, ph * ratio, ph * ratio)
            .unfold(3, pw * ratio, pw * ratio)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(-1, inp.shape[1], ph * ratio, pw * ratio)
        )

        target_blocks = (
            target.unfold(2, ph, ph)
            .unfold(3, pw, pw)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(-1, C_target, ph, pw)
        )

        matched_blocks = block_mapping(inp_blocks, target_blocks)

        pan_match = matched_blocks.reshape(
            inp.shape[0], num_h_blocks, num_w_blocks, C_target, ph * ratio, pw * ratio
        )
        pan_match = pan_match.permute(0, 3, 1, 4, 2, 5)
        pan_match = pan_match.reshape(
            inp.shape[0], C_target, h_ms * ratio, w_ms * ratio
        )

    return pan_match

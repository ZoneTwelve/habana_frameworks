###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import torch
from habana_frameworks.torch import _hpu_C
from habana_frameworks.torch.dynamo.compile_backend.decomposition import override_decomposition_table
from torch._decomp import global_decomposition_table
from torch._meta_registrations import _compute_reduction_shape, register_meta, utils
from torch._ops import HigherOrderOperator, OpOverload

_meta_lib_dont_use_me_use_register_meta_for_hpu = torch.library.Library("hpu", "IMPL", "Meta")

# If non-trivial shape calculation is necessary call C++ code instead of copying
# similar calculation algorithm in python
# See:
# custom_op_calc_out_shape*
# in python and
# REGISTER_CUSTOM_OP_OUTSHAPE_FUN
# in C++ code


@register_meta([torch.ops.hpu.instance_norm.default])
def instance_norm(input, weight_opt, bias_opt, eps):
    out = torch.empty_like(input)
    mean_tensor = input.new_empty((input.shape[0], input.shape[1]), dtype=torch.float32)
    istd_tensor = input.new_empty((input.shape[0], input.shape[1]), dtype=torch.float32)
    return [out, mean_tensor, istd_tensor]


@register_meta([torch.ops.hpu.instance_norm_backward.default])
def instance_norm_bwd(input, grad_in, mean, istd, gamma):
    out = torch.empty_like(input)
    grad_beta_tensor = input.new_empty((input.shape[1]), dtype=torch.float32)
    grad_gamma_tensor = input.new_empty((input.shape[1]), dtype=torch.float32)
    return [out, grad_beta_tensor, grad_gamma_tensor]


@register_meta([torch.ops.hpu.cast_to_fp8.default])
def meta_cast_to_fp8(input, scale, stochastic, out, amax):
    return out, amax


def meta_cast_to_fp8_v2_common(input, is_amax, dtype):
    out_dtype = dtype if dtype else torch.int8
    out = input.new_empty(input.shape, dtype=out_dtype)
    amax_shape = () if is_amax else 0
    amax = input.new_empty(amax_shape, dtype=torch.float32)
    return out, amax


@register_meta([torch.ops.hpu.cast_to_fp8_v2.default])
def meta_cast_to_fp8_v2(input, scale=None, stochastic=False, is_amax=False, dtype=None, scale_shape=None):
    return meta_cast_to_fp8_v2_common(input, is_amax, dtype)


@register_meta([torch.ops.hpu.cast_to_fp8_v2.scalar])
def meta_cast_to_fp8_v2_scalar(input, scale, stochastic=False, is_amax=False, dtype=None, scale_shape=None):
    return meta_cast_to_fp8_v2_common(input, is_amax, dtype)


@register_meta([torch.ops.hpu.cast_to_fp8_v2.scalar_list])
def meta_cast_to_fp8_v2_scalar_list(input, scale, stochastic=False, is_amax=False, dtype=None, scale_shape=None):
    return meta_cast_to_fp8_v2_common(input, is_amax, dtype)


@register_meta([torch.ops.hpu.cast_to_fp8_hybrid.default])
def meta_cast_to_fp8_hybrid(input, scale_152=None, scale_143=None, stochastic=False, is_amax=False):
    out_152 = input.new_empty(input.shape, dtype=torch.float8_e5m2)
    out_143 = input.new_empty(input.shape, dtype=torch.float8_e4m3fn)
    amax_shape = () if is_amax else 0
    amax = input.new_empty(amax_shape, dtype=torch.float32)
    return out_152, out_143, amax


def meta_convert_from_int4_common(input, out_dtype):
    output_shape = list(input.shape)
    output_shape[-1] *= 8
    return input.new_empty(output_shape, dtype=out_dtype)


@register_meta([torch.ops.hpu.convert_from_int4.default])
def meta_convert_from_int4(input, scale, zero_point, out_dtype):
    return meta_convert_from_int4_common(input, out_dtype)


@register_meta([torch.ops.hpu.convert_from_uint4.default])
def meta_convert_from_uint4(input, scale, zero_point, out_dtype):
    return meta_convert_from_int4_common(input, out_dtype)


@register_meta([torch.ops.hpu.fp8_cast_transpose.default])
def meta_fp8_cast_transpose(input, scale, stochastic, out, transposed, amax):
    return out, transposed, amax


@register_meta([torch.ops.hpu.fp8_cast_transpose_bgrad.default])
def meta_fp8_cast_transpose_bgrad(input, scale, stochastic, out, transposed, bgrad, amax):
    return out, transposed, bgrad, amax


@register_meta([torch.ops.hpu.fp8_cast_transpose_bgrad_dgelu.default])
def meta_fp8_cast_transpose_bgrad_dgelu(grad, input, scale, retain, stochastic, out, transposed, bgrad, amax):
    return out, transposed, bgrad, amax


@register_meta([torch.ops.hpu.cast_from_fp8.default])
def meta_cast_from_fp8(input, scale, out_dtype, scale_shape=None):
    return input.new_empty(input.shape, dtype=out_dtype)


@register_meta([torch.ops.hpu.cast_from_fp8.scalar])
def meta_cast_from_fp8_scalar(input, scale, out_dtype, scale_shape=None):
    return input.new_empty(input.shape, dtype=out_dtype)


@register_meta([torch.ops.hpu.cast_from_fp8.scalar_list])
def meta_cast_from_fp8_scalar_list(input, scale, out_dtype, scale_shape=None):
    return input.new_empty(input.shape, dtype=out_dtype)


@register_meta([torch.ops.hpu.fp8_dropout.default])
def meta_fp8_dropout(input, p, scale, stochastic_rounding, is_amax, dtype):
    out_dtype = dtype if dtype else torch.int8
    out = input.new_empty(input.shape, dtype=out_dtype)
    mask = input.new_empty(input.shape, dtype=torch.int8)
    amax = input.new_empty((), dtype=torch.float32)
    return out, mask, amax


@register_meta([torch.ops.hpu.fp8_gelu.default])
def meta_fp8_gelu(input, scale, stochastic, out, retain, amax):
    return out, retain, amax


@register_meta([torch.ops.hpu.fp8_bgrad_dgelu.default])
def meta_fp8_bgrad_dgelu(grad, input, scale, retain, stochastic, is_amax, dtype):
    out_dtype = dtype if dtype else torch.int8
    out = input.new_empty(input.shape, dtype=out_dtype)
    bgrad = input.new_empty(input.shape[1], dtype=input.dtype)
    amax = input.new_empty((), dtype=torch.float32)
    return out, bgrad, amax


@register_meta([torch.ops.hpu.fp8_fast_softmax.default])
def meta_fp8_fast_softmax(input, mask, scale, softmax_scale, stochastic, is_amax, dtype):
    out_dtype = dtype if dtype else torch.int8
    out = input.new_empty(input.shape, dtype=out_dtype)
    amax = input.new_empty((), dtype=torch.float32)
    return out, amax


@register_meta([torch.ops.hpu.fp8_gelu_v2.default])
def meta_fp8_gelu_v2(input, scale, stochastic, is_amax, dtype):
    out_dtype = dtype if dtype else torch.int8
    out = input.new_empty(input.shape, dtype=out_dtype)
    retain = input.new_empty(input.shape)
    amax = input.new_empty((), dtype=torch.float32)
    return out, retain, amax


@register_meta([torch.ops.hpu.fp8_layernorm.default])
def meta_fp8_layernorm(input, weight, bias, eps, scale, stochastic, out, mean, istd, amax):
    return out, mean, istd, amax


@register_meta([torch.ops.hpu.fp8_gemm.default])
def meta_fp8_gemm(
    A,
    trans_A,
    B,
    trans_B,
    D,
    out_dtype,
    A_scale_inv,
    B_scale_inv,
    bias,
    accumulate,
    out,
):
    return out


def meta_fp8_gemm_v2_common(
    A,
    trans_A,
    B,
    trans_B,
    out_dtype,
):
    out_shape = _hpu_C.custom_op_calc_out_shape_params_int("fp8_gemm_v2", [A, B], [trans_A, trans_B])[0]
    out = A.new_empty(out_shape, dtype=out_dtype)
    return out


@register_meta([torch.ops.hpu.fp8_gemm_v2.default])
def meta_fp8_gemm_v2(
    A,
    trans_A,
    B,
    trans_B,
    D,
    out_dtype,
    A_scale_inv=None,
    B_scale_inv=None,
    bias=None,
    accumulate=False,
    scale_shape=None,
):
    return meta_fp8_gemm_v2_common(A, trans_A, B, trans_B, out_dtype)


@register_meta([torch.ops.hpu.fp8_gemm_v2.scalar])
def meta_fp8_gemm_v2_scalar(
    A,
    trans_A,
    B,
    trans_B,
    D,
    out_dtype,
    A_scale_inv,
    B_scale_inv,
    bias=None,
    accumulate=False,
    scale_shape=None,
):
    return meta_fp8_gemm_v2_common(A, trans_A, B, trans_B, out_dtype)


@register_meta([torch.ops.hpu.fp8_gemm_v2.scalar_list])
def meta_fp8_gemm_v2_scalar_list(
    A,
    trans_A,
    B,
    trans_B,
    D,
    out_dtype,
    A_scale_inv,
    B_scale_inv,
    bias=None,
    accumulate=False,
    scale_shape=None,
):
    return meta_fp8_gemm_v2_common(A, trans_A, B, trans_B, out_dtype)


def to_list_if_necessary(input, size):
    return input if hasattr(input, "__iter__") else [input] * size


def meta_conv2d_fp8_common(
    input,
    weight,
    stride=1,
    padding=0,
    dilation=1,
    out_dtype=None,
):
    stride = to_list_if_necessary(stride, 2)
    padding = to_list_if_necessary(padding, 2)
    dilation = to_list_if_necessary(dilation, 2)
    out_shape = _hpu_C.custom_op_calc_out_shape_params_int("conv2d_fp8", [input, weight], stride + padding + dilation)[
        0
    ]

    output_dtype = out_dtype if out_dtype else torch.bfloat16
    return input.new_empty(out_shape, dtype=output_dtype)


@register_meta([torch.ops.hpu.conv2d_fp8.default])
def meta_conv2d_fp8(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    out_dtype=None,
    scale_input=None,
    scale_weight=None,
):
    return meta_conv2d_fp8_common(input, weight, stride, padding, dilation, out_dtype)


@register_meta([torch.ops.hpu.conv2d_fp8.scalar])
def meta_conv2d_fp8_scalar(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    out_dtype=None,
    scale_input=None,
    scale_weight=None,
):
    return meta_conv2d_fp8_common(input, weight, stride, padding, dilation, out_dtype)


@register_meta([torch.ops.hpu.fp8_transpose.default])
def meta_fp8_transpose(input, dims, out):
    return out


@register_meta([torch.ops.hpu.fp8_permute.default])
def meta_fp8_permute(input, out):
    return out


@register_meta([torch.ops.hpu.fp8_reshape.default])
def meta_fp8_reshape(input, shape):
    return input.new_empty(shape)


@register_meta([torch.ops.hpu.optimizer_lamb_fused_norm.default])
def meta_optimizer_lamb_fused_norm(grads, scale):
    return grads[0].new_empty((1,))


@register_meta([torch.ops.hpu.optimizer_resource_apply_momentum.default])
def meta_optimizer_resource_apply_momentum(params_momentum_buf_list, dp_list, momentum):
    return


@register_meta([torch.ops.hpu.optimizer_lars.default])
def meta_optimizer_optimizer_lars(params, grads, skip_masks, eeta, weight_decay, eps, lr):
    return


@register_meta([torch.ops.hpu.optimizer_lamb_phase2.default])
def meta_optimizer_lamb_phase2(weights, adam_norms, weight_norms, adam_steps, step, weight_decay, use_lamb):
    return


@register_meta([torch.ops.hpu.optimizer_ema.default])
def meta_optimizer_optimizer_ema(model_inputs, updated_ema, decay):
    return


@register_meta([torch.ops.hpu.optimizer_adamw.default])
def meta_optimizer_adamw(
    gradient_vec,
    weight_vec,
    exp_avg_vec,
    exp_avg_sq_vec,
    lr,
    neg_step_t,
    beta1,
    beta2,
    epsilon,
    weight_decay,
    has_weight_decay,
):
    return


@register_meta([torch.ops.hpu.masked_batch_gemm.default])
def meta_masked_batch_gemm(a, b, mask_a, mask_b, trans_a, trans_b):
    out_shape = _hpu_C.custom_op_calc_out_shape_params_int("masked_batch_gemm", [a, b], [trans_a, trans_b])[0]
    out = a.new_empty(out_shape)
    return out


@register_meta([torch.ops.hpu.scaled_triangular_softmax.default])
def meta_scaled_triangular_softmax(input, inv_scale_attn, exp_sum_recpr=None, sum=None):
    return input.new_empty(input.shape)


@register_meta([torch.ops.hpu.scaled_triangular_softmax_retain.default])
def meta_scaled_triangular_softmax_retain(input, inv_scale_attn):
    out_shape = input.shape
    retain_shape = out_shape[:-1] + (1,)
    out = input.new_empty(out_shape)
    exp_sum_recpr = input.new_empty(retain_shape, dtype=torch.float32)
    max = input.new_empty(retain_shape)
    return out, exp_sum_recpr, max


@register_meta([torch.ops.hpu.fp8_copy_.default])
def meta_fp8_copy_(self, src):
    return self


@register_meta([torch.ops.hpu.fp8_kv_reorder_.default])
def meta_fp8_kv_reorder_(self, start, end, beam_idx):
    return self


@register_meta([torch.ops.hpu.kv_reorder_.default])
def meta_kv_reorder_(self, start, end, beam_idx):
    return self


@register_meta([torch.ops.hpu.kv_reorder.default])
def meta_kv_reorder(self, start, end, beam_idx):
    return self.new_empty(self.shape)


@register_meta([torch.ops.hpu.fp8_index_copy_.default])
def meta_fp8_index_copy_(self, src):
    return self


@register_meta([torch.ops.hpu.fp8_repeat_v2.default])
def meta_fp8_repeat_v2(self, repeats):
    if len(repeats) == 0:
        return self

    out_shape = _hpu_C.custom_op_calc_out_shape_params_int("fp8_repeat_v2", [self], repeats)[0]
    return self.new_empty(out_shape)


@register_meta([torch.ops.hpu.fp8_index_select_v2.default])
def meta_fp8_index_select_v2(self, dim, index):
    result_size = list(self.size())
    if self.dim() > 0:
        result_size[dim] = index.numel()
    return self.new_empty(result_size)


@register_meta([torch.ops.hpu.scaled_masked_triangular_softmax.default])
def meta_scaled_masked_triangular_softmax(
    self,
    start_end,
    inv_scale_attn,
    grouped_batch_size,
    use_max,
    mode,
    out_dtype=None,
):
    dtype = out_dtype if out_dtype else self.dtype
    return self.new_empty(self.shape, dtype=dtype)


def meta_sdpa_recomp_fwd_helper(q, k, v, requires_backward):
    seed_dtype = torch.int

    out_shapes = _hpu_C.custom_op_calc_out_shape_params_int("sdpa_recomp_fwd", [q, k, v], [requires_backward])
    out_tensors = [q.new_empty(s) for s in out_shapes[:-1]]
    out_tensors.append(q.new_empty(out_shapes[-1], dtype=seed_dtype))

    return out_tensors


@register_meta([torch.ops.hpu.sdpa_recomp_fwd.default])
def meta_sdpa_recomp_fwd(
    q, k, v, attn_mask, dropout_p, is_causal, scale, requires_backward, softmax_mode, valid_seq_len, seq_padding_type
):
    return meta_sdpa_recomp_fwd_helper(q, k, v, requires_backward)


@register_meta([torch.ops.hpu.sdpa_recomp_fwd_dropout.default])
def meta_sdpa_recomp_fwd_dropout(
    q, k, v, attn_mask, dropout_p, is_causal, scale, requires_backward, softmax_mode, valid_seq_len, seq_padding_type
):
    return meta_sdpa_recomp_fwd_helper(q, k, v, requires_backward)


@register_meta([torch.ops.hpu.sdpa_recomp_fwd_non_dropout.default])
def meta_sdpa_recomp_fwd_non_dropout(
    q, k, v, attn_mask, dropout_p, is_causal, scale, requires_backward, softmax_mode, valid_seq_len, seq_padding_type
):
    return meta_sdpa_recomp_fwd_helper(q, k, v, requires_backward)


@register_meta([torch.ops.hpu.sdpa_recomp_fwd_dropout_seed.default])
def meta_sdpa_recomp_fwd_dropout_seed(
    seed,
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    requires_backward,
    softmax_mode,
    valid_seq_len,
    seq_padding_type,
):
    return meta_sdpa_recomp_fwd_helper(q, k, v, requires_backward)


@register_meta([torch.ops.hpu.sdpa_recomp_bwd.default])
def meta_sdpa_recomp_bwd(dout, q, k, v, attn_mask, m, linv, seed, is_causal, dropout_p, scale, fast_softmax_mode):
    grad_q = q.new_empty(q.shape)
    grad_k = k.new_empty(k.shape)
    grad_v = v.new_empty(v.shape)
    return grad_q, grad_k, grad_v


def meta_sdpa_fwd_helper(q, k, v, dropout_p):
    out_shapes = _hpu_C.custom_op_calc_out_shape_params_float("sdpa_fwd", [q, k, v], [dropout_p])
    out_tensors = [q.new_empty(s) for s in out_shapes[:-1]]
    out_tensors.append(q.new_empty(out_shapes[-1], dtype=torch.int8))
    return out_tensors


@register_meta([torch.ops.hpu.sdpa_fwd.default])
def meta_sdpa_fwd(q, k, v, attn_mask, dropout_p, scale, is_causal, fast_softmax_mode, valid_seq_len, seq_padding_type):
    return meta_sdpa_fwd_helper(q, k, v, dropout_p)


@register_meta([torch.ops.hpu.sdpa_fwd_dropout.default])
def meta_sdpa_fwd_dropout(
    q, k, v, attn_mask, dropout_p, scale, is_causal, fast_softmax_mode, valid_seq_len, seq_padding_type
):
    return meta_sdpa_fwd_helper(q, k, v, dropout_p)


@register_meta([torch.ops.hpu.sdpa_fwd_non_dropout.default])
def meta_sdpa_fwd_non_dropout(
    q, k, v, attn_mask, dropout_p, scale, is_causal, fast_softmax_mode, valid_seq_len, seq_padding_type
):
    return meta_sdpa_fwd_helper(q, k, v, dropout_p)


@register_meta([torch.ops.hpu.sdpa_fwd_dropout_seed.default])
def meta_sdpa_fwd_dropout_seed(
    seed, q, k, v, attn_mask, dropout_p, scale, is_causal, fast_softmax_mode, valid_seq_len, seq_padding_type
):
    return meta_sdpa_fwd_helper(q, k, v, dropout_p)


@register_meta([torch.ops.hpu.sdpa_bwd.default])
def meta_sdpa_bwd(dout, q, k, v, p, dm, is_causal, dropout_p, scale):
    grad_q = q.new_empty(q.shape)
    grad_k = k.new_empty(k.shape)
    grad_v = v.new_empty(v.shape)
    return grad_q, grad_k, grad_v


@register_meta([torch.ops.hpu.softmax_fp8.default])
def meta_softmax_fp8(input, dim, input_scale=None, output_scale=None, inv_attn_heads=None, fused_add=None):
    if input_scale is None:
        dtype = torch.bfloat16
    else:
        dtype = torch.float8_e4m3fn
    return input.new_empty(input.shape, dtype=dtype)


@register_meta([torch.ops.hpu.in_place_interleave_.default])
def meta_in_place_interleave_(self):
    return self


@register_meta([torch.ops.hpu.in_place_interleave.default])
def meta_in_place_interleave(self):
    return self.new_empty(self.shape)


@register_meta([torch.ops.hpu.custom_softmax.default])
def meta_custom_softmax(input, flavor):
    return input.new_empty(input.shape)


@register_meta([torch.ops.hpu.rotary_pos_embedding.default])
def meta_rotary_pos_embedding(input, sin, cos, position_ids, offset, mode):
    return input.new_empty(input.shape)


@register_meta([torch.ops.hpu.rotary_pos_embedding_backward.default])
def meta_rotary_pos_embedding_backward(grad_in, sin, cos, position_ids, offset, mode):
    return grad_in.new_empty(grad_in.shape)


@register_meta([torch.ops.hpu.rms_norm.default, torch.ops.hpu.rms_norm_fast.default])
def meta_rms_norm(data_in, gamma, epsilon):
    inverse_root_mean_square_shape = list(data_in.shape)
    inverse_root_mean_square_shape[-1] = 1

    data_in_dtype = data_in.dtype
    if data_in_dtype != gamma.dtype:
        data_in_dtype = torch.float32

    return data_in.new_empty(data_in.shape, dtype=data_in_dtype), data_in.new_empty(
        inverse_root_mean_square_shape, dtype=torch.float32
    )


@register_meta([torch.ops.hpu.rms_norm_backward.default])
def meta_rms_norm_backward(grad_in, data_in, gamma, inverse_rms, use_stages, bwd_mode):
    return data_in.new_empty(data_in.shape), gamma.new_empty(gamma.shape)


@register_meta([torch.ops.hpu.ctc_loss_custom.default])
def meta_ctc_loss_custom(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity):
    loss_shape, alpha_shape = _hpu_C.custom_op_calc_out_shape_params_int(
        "ctc_loss_custom", [log_probs, targets], [reduction]
    )
    return input_lengths.new_empty(loss_shape, dtype=log_probs.dtype), log_probs.new_empty(alpha_shape)


@register_meta([torch.ops.hpu.ctc_loss_custom_backward.default])
def meta_ctc_loss_custom_backward(
    loss_grad_in, log_probs, targets, input_lengths, target_lengths, loss, alpha, blank, reduction, zero_infinity
):
    return log_probs.new_empty(log_probs.shape)


@register_meta([torch.ops.hpu.sum_fp8.default])
def meta_sum_fp8(self, dim=None, keepdim=False, out_dtype=None):
    dim = utils.reduction_dims(self.shape, dim)
    output_shape = _compute_reduction_shape(self, dim, keepdim)
    output_dtype = out_dtype if out_dtype else self.dtype
    return self.new_empty(output_shape, dtype=output_dtype)


def activate_hpu_custom_op_meta():
    activate_meta_table = {}

    # For a given op, we pick the most specific decomp function from
    # global_decomp_table in the precedence order of meta > post_autograd > pre_autograd
    for type in ["meta", "post_autograd", "pre_autograd"]:
        registry = global_decomposition_table[type]

        for opo in registry:
            if opo not in activate_meta_table:
                activate_meta_table[opo] = registry[opo]

    for op_overload, fn in activate_meta_table.items():
        if isinstance(op_overload, HigherOrderOperator):
            continue
        assert isinstance(op_overload, OpOverload)

        if "hpu::" not in op_overload.name():
            continue

        op_overload.py_impl(torch._C.DispatchKey.Meta)(fn)

        _meta_lib_dont_use_me_use_register_meta_for_hpu.impl(op_overload, fn)

    # Need to update the py_kernel registriation for _to_copy since we have
    # a new version _to_copy registered.
    # This should be removed if upstream pytorch included fix for:
    # https://github.com/pytorch/pytorch/issues/128202
    for op_overload, fn in override_decomposition_table.items():
        if op_overload.has_kernel_for_dispatch_key(torch._C.DispatchKey.Meta):
            op_overload.py_kernels.pop(torch._C.DispatchKey.Meta, None)
        op_overload.py_impl(torch._C.DispatchKey.Meta)(fn)


activate_hpu_custom_op_meta()

import torch
from torch.nn import functional as F
import numpy as np

# 默认约束参数，防止数值不稳定（比如除 0 或梯度爆炸）
DEFAULT_MIN_BIN_WIDTH = 1e-3       # bin 最小宽度
DEFAULT_MIN_BIN_HEIGHT = 1e-3      # bin 最小高度
DEFAULT_MIN_DERIVATIVE = 1e-3      # 最小导数（避免变换函数太平或不光滑）

def piecewise_rational_quadratic_transform(
    inputs,                          # 输入张量（通常是 z）
    unnormalized_widths,            # 未归一化的 bin 宽度参数（待变换）
    unnormalized_heights,           # 未归一化的 bin 高度参数
    unnormalized_derivatives,       # 未归一化的导数（控制 spline 曲率）
    inverse=False,                  # 是否执行逆向变换（用于采样）
    tails=None,                     # 是否添加尾部处理（None 或 "linear"）
    tail_bound=1.0,                 # 尾部截断范围（仅当 tails 不为 None 时生效）
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,         # 控制最小 bin 宽度（避免除零）
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,       # 控制最小 bin 高度
    min_derivative=DEFAULT_MIN_DERIVATIVE        # 控制最小导数
):
    """
    对输入进行分段有理二次变换（Piecewise Rational Quadratic Spline），可用于 Flow 模块中建模复杂可逆变换。

    Returns:
        outputs: 变换后的值（同样形状）
        logabsdet: 对应的对数行列式（Jacobian，用于 Flow 的 NLL）
    """
    # 如果没有 tails，使用标准的 rational_quadratic_spline（只作用于 [-1, 1] 区间内）
    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        # 如果开启 tails，使用扩展版本（例如 tails="linear"，处理尾部为线性变换）
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    # 执行 spline 函数（前向或反向）
    outputs, logabsdet = spline_fn(
        inputs=inputs,                                      # 输入张量（通常为 [B, C, T]）
        unnormalized_widths=unnormalized_widths,            # spline 的宽度参数
        unnormalized_heights=unnormalized_heights,          # spline 的高度参数
        unnormalized_derivatives=unnormalized_derivatives,  # spline 的导数参数
        inverse=inverse,                                    # 是否反向采样
        min_bin_width=min_bin_width,                        # 最小宽度限制
        min_bin_height=min_bin_height,                      # 最小高度限制
        min_derivative=min_derivative,                      # 最小导数限制
        **spline_kwargs                                     # 可能包含 tails 和 tail_bound
    )

    return outputs, logabsdet


def searchsorted(bin_locations, inputs, eps=1e-6):
    # 为了确保 inputs 落入最后一个 bin 时不会越界，手动将最后一个 bin 的右边界稍微抬高一点
    bin_locations[..., -1] += eps

    # 将 inputs 插入到 bin 区间中，判断它属于哪个 bin：
    # 对于每个 input，在所有 bin 边界中找到第一个大于 input 的位置
    # torch.sum(inputs[..., None] >= bin_locations, dim=-1) 会返回 input 落入的 bin 编号（从 1 开始），所以减 1 得到 0-based 下标
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=False,
        tails="linear",
        tail_bound=1.0,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    # 创建掩码，标记哪些输入值位于 [-tail_bound, +tail_bound] 区间内
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask  # 其余部分为边界之外

    # 初始化输出值和 log Jacobian 行列式
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        # 对边界之外的 spline 设置线性映射（identity function）

        # 在最前面和最后面补上一个导数值（用于 spline 衔接边界）
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))

        # 设置边界的导数常量，确保平滑连接（避免爆炸梯度）
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant  # 左边界导数
        unnormalized_derivatives[..., -1] = constant  # 右边界导数

        # 区间外不做变换，输出为输入本身，log det 为 0
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        # 如果使用了未实现的边界处理方式，则抛出异常
        raise RuntimeError("{} tails are not implemented.".format(tails))

    # 对区间内的部分应用 rational quadratic spline 映射
    (
        outputs[inside_interval_mask],
        logabsdet[inside_interval_mask],
    ) = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],  # 当前 batch 中在区间内的输入
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,  # spline 左边界
        right=tail_bound,  # spline 右边界
        bottom=-tail_bound,  # y 方向下边界
        top=tail_bound,  # y 方向上边界
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    # 检查输入是否超出定义域范围
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    # 检查最小宽高设置是否合理
    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    # ----------- Step 1: 宽度归一化并映射到实际 bin 宽度 ------------
    widths = F.softmax(unnormalized_widths, dim=-1)  # 归一化权重
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths  # 加上最小宽度限制
    cumwidths = torch.cumsum(widths, dim=-1)  # 累积得到分段边界
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left  # 映射到 [left, right]
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]  # 实际每段宽度

    # ----------- Step 2: 高度归一化并映射到实际 bin 高度 ------------
    derivatives = min_derivative + F.softplus(unnormalized_derivatives)  # softplus 保证正值

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    # ----------- Step 3: 计算每个输入所属的 bin ------------
    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    # ----------- Step 4: 提取当前 bin 的边界信息 ------------
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    # ----------- Step 5: Inverse 方向（从 y 推 x） ------------
    if inverse:
        # 计算三次方程的系数 a*x^2 + b*x + c = 0
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)

        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )

        c = -input_delta * (inputs - input_cumheights)

        # 解一元二次方程
        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))  # 数值稳定版
        outputs = root * input_bin_widths + input_cumwidths

        # 计算 log|dy/dx|，用于 flow 的 log likelihood
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet

    # ----------- Step 6: Forward 方向（从 x 推 y） ------------
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths  # 将输入归一化到当前 bin 内
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        # 计算 log|dy/dx|
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet


import math


def lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:

    if t < T_w:
        return (t / T_w) * alpha_max

    if t <= T_c:
        cos_term = math.cos(math.pi * (t - T_w) / (T_c - T_w))
        return alpha_min + 0.5 * (1 + cos_term) * (alpha_max - alpha_min)

    return alpha_min
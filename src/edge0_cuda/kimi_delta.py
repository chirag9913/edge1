import torch
import torch.nn.functional as F


class ShortConvolution:
    def __init__(self, weight, activation="silu"):
        self.weight = weight.float()
        self.activation = activation
        self.kernel_size = weight.shape[-1]

    def __call__(self, x, state=None):
        b, t, c = x.shape

        if state is None:
            state = torch.zeros(
                b,
                c,
                self.kernel_size - 1,
                dtype=x.dtype,
                device=x.device,
            )

        x_t = x.transpose(1, 2)

        x_full = torch.cat(
            [state.to(x_t.dtype), x_t],
            dim=-1,
        )

        y = F.conv1d(
            x_full,
            self.weight.to(x_t.dtype),
            groups=c,
        )

        new_state = x_full[:, :, -(self.kernel_size - 1):]

        y = y.transpose(1, 2)

        if self.activation == "silu":
            y = F.silu(y)

        return y, new_state


class FusedRMSNormGated:
    def __init__(self, weight, eps=1e-6, activation="sigmoid"):
        self.weight = weight.float()
        self.eps = eps
        self.activation = activation

    def __call__(self, x, gate):
        if self.activation == "sigmoid":
            gate = torch.sigmoid(gate)
        elif self.activation == "silu":
            gate = F.silu(gate)

        x = x.float()

        variance = x.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        x = x * torch.rsqrt(
            variance + self.eps
        )

        x = x * self.weight

        return x * gate.float()


def l2_normalize(x, dim=-1, eps=1e-6):
    return x / torch.linalg.vector_norm(
        x,
        dim=dim,
        keepdim=True,
    ).clamp_min(eps)


class KDARecurrent:
    def __init__(
        self,
        A_log,
        dt_bias,
        num_heads,
        head_dim,
    ):
        self.A_log = A_log.float()
        self.dt_bias = dt_bias.float()
        self.num_heads = num_heads
        self.head_dim = head_dim

        if self.A_log.numel() != num_heads:
            raise ValueError(
                f"A_log must have {num_heads} values"
            )

        if self.dt_bias.numel() != num_heads * head_dim:
            raise ValueError(
                f"dt_bias must have {num_heads * head_dim} values"
            )

    def __call__(
        self,
        q,
        k,
        v,
        beta,
        state=None,
    ):
        b, t, heads, dim = q.shape

        if state is None:
            state = torch.zeros(
                b,
                heads,
                dim,
                dim,
                dtype=torch.float32,
                device=q.device,
            )

        A = -torch.exp(
            self.A_log
        ).view(
            1,
            heads,
            1,
            1,
        )

        dt = F.softplus(
            self.dt_bias
        ).view(
            1,
            heads,
            dim,
        )

        state = state.float()
        outputs = []

        decay = torch.exp(
            A * dt.unsqueeze(-1)
        )

        for i in range(t):
            qi = q[:, i].float()
            ki = k[:, i].float()
            vi = v[:, i].float()
            bi = beta[:, i].float()

            state = state * decay

            kv = torch.einsum(
                "bhm,bhn->bhmn",
                ki,
                vi,
            )

            state = state + (
                bi[:, :, None, None] * kv
            )

            out = torch.einsum(
                "bhm,bhmn->bhn",
                qi,
                state,
            )

            outputs.append(out)

        return torch.stack(
            outputs,
            dim=1,
        ), state


class KimiDeltaAttention:
    def __init__(
        self,
        q_proj,
        k_proj,
        v_proj,
        q_conv,
        k_conv,
        v_conv,
        f_proj,
        b_proj,
        g_proj,
        o_norm,
        o_proj,
        A_log,
        dt_bias,
        num_heads=16,
        head_dim=128,
        rms_norm_eps=1e-6,
        state_heads=None,
    ):
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj

        self.q_conv = ShortConvolution(q_conv)
        self.k_conv = ShortConvolution(k_conv)
        self.v_conv = ShortConvolution(v_conv)

        self.f_proj = f_proj
        self.b_proj = b_proj
        self.g_proj = g_proj

        self.o_norm = FusedRMSNormGated(
            o_norm,
            eps=rms_norm_eps,
        )

        self.o_proj = o_proj

        self.num_heads = num_heads
        self.head_dim = head_dim

        if state_heads is None:
            state_heads = A_log.numel()

        self.state_heads = state_heads

        if num_heads % state_heads != 0:
            raise ValueError(
                "num_heads must be divisible by state_heads"
            )

        if dt_bias.numel() != state_heads * head_dim:
            raise ValueError(
                f"dt_bias must contain "
                f"{state_heads * head_dim} values"
            )

        self.head_repeat = num_heads // state_heads

        self.kda = KDARecurrent(
            A_log,
            dt_bias,
            state_heads,
            head_dim,
        )

    def _empty_conv_state(self, x, conv):
        return torch.zeros(
            x.shape[0],
            x.shape[-1],
            conv.kernel_size - 1,
            dtype=x.dtype,
            device=x.device,
        )

    @torch.no_grad()
    def __call__(
        self,
        hidden_states,
        initial_state=None,
        return_state=False,
    ):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if initial_state is None:
            conv_q_state = None
            conv_k_state = None
            conv_v_state = None
            kda_state = None
        else:
            conv_q_state = initial_state["q_conv"]
            conv_k_state = initial_state["k_conv"]
            conv_v_state = initial_state["v_conv"]
            kda_state = initial_state["kda"]

        q, next_q_conv = self.q_conv(
            q,
            conv_q_state,
        )

        k, next_k_conv = self.k_conv(
            k,
            conv_k_state,
        )

        v, next_v_conv = self.v_conv(
            v,
            conv_v_state,
        )

        b, t, _ = q.shape

        q = q.reshape(
            b,
            t,
            self.num_heads,
            self.head_dim,
        )

        k = k.reshape(
            b,
            t,
            self.num_heads,
            self.head_dim,
        )

        v = v.reshape(
            b,
            t,
            self.num_heads,
            self.head_dim,
        )

        q = l2_normalize(q, dim=-1)
        k = l2_normalize(k, dim=-1)

        if self.state_heads != self.num_heads:
            q_state = q.reshape(
                b,
                t,
                self.state_heads,
                self.head_repeat,
                self.head_dim,
            ).mean(dim=3)

            k_state = k.reshape(
                b,
                t,
                self.state_heads,
                self.head_repeat,
                self.head_dim,
            ).mean(dim=3)

            v_state = v.reshape(
                b,
                t,
                self.state_heads,
                self.head_repeat,
                self.head_dim,
            ).mean(dim=3)
        else:
            q_state = q
            k_state = k
            v_state = v

        beta = torch.sigmoid(
            self.b_proj(hidden_states)
        )

        beta = beta.reshape(
            b,
            t,
            self.num_heads,
        )

        if self.state_heads != self.num_heads:
            beta = beta.reshape(
                b,
                t,
                self.state_heads,
                self.head_repeat,
            ).mean(dim=-1)

        out, next_kda_state = self.kda(
            q_state,
            k_state,
            v_state,
            beta,
            kda_state,
        )

        if self.state_heads != self.num_heads:
            out = out.unsqueeze(3).expand(
                b,
                t,
                self.state_heads,
                self.head_repeat,
                self.head_dim,
            ).reshape(
                b,
                t,
                self.num_heads,
                self.head_dim,
            )

        gate = self.g_proj(hidden_states)

        gate = gate.reshape(
            b,
            t,
            self.num_heads,
            self.head_dim,
        )

        out = self.o_norm(
            out,
            gate,
        )

        out = out.reshape(
            b,
            t,
            self.num_heads * self.head_dim,
        )

        out = self.o_proj(out)

        next_state = {
            "q_conv": next_q_conv,
            "k_conv": next_k_conv,
            "v_conv": next_v_conv,
            "kda": next_kda_state,
        }

        if return_state:
            return out, next_state

        return out

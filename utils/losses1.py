import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- core: scoring rules ----------
def scoring_matrix(p: torch.Tensor, loss_code: str, eps: float = 1e-8) -> torch.Tensor:
    """
    Given class probabilities p (B, C), return S(p, y=c) for all c as a matrix (B, C).
    Each column c is the loss value if the true label is c.
    """
    p = p.clamp_min(eps)                     # avoid log/zero issues
    B, C = p.shape

    if loss_code == "cross_entropy":
        # S(p,y=c) = -log p_c
        return -torch.log(p)

    elif loss_code in ("brier", "squared", "mse"):
        # S(p,y=c) = (p - e_c)^2 summed over classes = 1 - 2 p_c + sum_j p_j^2
        sumsq = (p * p).sum(dim=1, keepdim=True)          # (B,1)
        return 1.0 - 2.0 * p + sumsq                      # (B,C)

    elif loss_code == "spherical":
        # S(p,y=c) = - p_c / ||p||_2
        denom = p.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        return -p / denom

    elif loss_code.startswith("ps_"):  # pseudo-spherical: β > 1
        beta = float(loss_code.split("_", 1)[1])
        if beta <= 1:
            raise ValueError("pseudo-spherical requires beta > 1")
        # S(p,y=c) = - p_c^β / (β * ||p||_β)
        denom = (p.pow(beta).sum(dim=1, keepdim=True)).pow(1.0 / beta).clamp_min(eps)
        return -(p.pow(beta)) / (beta * denom)

    #elif loss_code.startswith("tsallis_"):  # Tsallis score: α ≠ 1
    #    alpha = float(loss_code.split("_", 1)[1])
    #    if abs(alpha - 1.0) < 1e-6:
            # limit α→1 equals log score
    #        return -torch.log(p)
        # One standard Tsallis (power) scoring rule:
        # S(p,y=c) = (p_c^{α-1} - sum_j p_j^α) / (α - 1)
    #    a = alpha - 1.0
    #    sum_pow = p.pow(alpha).sum(dim=1, keepdim=True)
    #   return (p.pow(alpha - 1.0) - sum_pow) / a

    elif loss_code.startswith("tsallis_"):  # Tsallis / power loss
        alpha = float(loss_code.split("_", 1)[1])
        if abs(alpha - 1.0) < 1e-6:
        # α→1 limit: log loss (proper)
            return -torch.log(p)

        # Proper Tsallis (power) loss from Bregman divergence:
        # L_α(p, y=c) = [1 - α p_c^{α-1} + (α - 1) * sum_j p_j^α] / (α - 1)
        a = alpha - 1.0
        sum_pow = p.pow(alpha).sum(dim=1, keepdim=True)  # sum_j p_j^α
        return (1.0 - alpha * p.pow(alpha - 1.0) + a * sum_pow) / a

    else:
        raise ValueError(f"Unknown proper loss code: {loss_code}")
    

# ---------- EM-style marginal-chain objective ----------
class MarginalChainProperLoss(nn.Module):
    """
    EM-style marginal-chain loss, 对齐 ForwardProperLoss（cross_entropy + M == F）:

      原始 MC:  L_MC = E_Q[-log p]
      Forward:  L_FW = -log (M p)_z

      数学上有：L_FW = L_MC - H(Q) - E_Q[log M_{z,·}]
      这里 Q 是 detach 的，所以 H(Q)、E_Q[log M] 对 logits 没梯度。
      我们在 cross_entropy 分支里显式减去这两项，让 loss 数值和 Forward 完全一致，
      但梯度保持不变。
    """
    def __init__(self, M, loss_code: str, reduction: str = "mean", eps: float = 1e-28):
        super().__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.M = torch.as_tensor(M, dtype=torch.float32)
        self.loss_code = loss_code
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = z.long()

        # p 和你原来的一样：log_softmax + exp
        logp = self.logsoftmax(logits)     # (B, C) = log p
        p = logp.exp()                     # (B, C) = p

        M = self.M.to(logits.device)       # (C, C)
        Mz = M[z]                          # (B, C) 取第 z 行

        # ---------- E-step: 责任 Q（数值上 posterior，反向里当常数） ----------
        numer = p.detach() * Mz            # (B, C)
        Q = numer / numer.sum(dim=1, keepdim=True).clamp_min(self.eps)
        Q = Q.detach()                     # 非常重要：Q 不反传梯度

        if self.loss_code == "cross_entropy":
            # ---------- 原始 MC 的部分：L_MC = E_Q[-log p] ----------
            # scoring_matrix(p, "cross_entropy") = -log(p)，
            # 这里直接用 logp 更干净：-log p = -logp.exp().log() = -logp
            S = -logp                      # (B, C)

            L_mc_per_sample = (Q * S).sum(dim=1)   # (B,)

            # ---------- 常数项：H(Q) = -sum Q log Q ----------
            Q_safe = Q.clamp_min(self.eps)
            H_Q = -(Q_safe * Q_safe.log()).sum(dim=1)   # (B,)

            # ---------- 常数项：E_Q[log M_{z,·}] ----------
            Mz_safe = Mz.clamp_min(self.eps)
            EQ_logM = (Q * Mz_safe.log()).sum(dim=1)    # (B,)

            # 对齐 Forward 的 loss：
            # L_FW = L_MC - H(Q) - E_Q[log M_{z,·}]
            loss_per_sample = L_mc_per_sample - H_Q - EQ_logM

        else:
            # ---------- 其它 scoring rule: 保持原始 MC 定义 ----------
            S = scoring_matrix(p, self.loss_code)       # (B, C)
            loss_per_sample = (Q * S).sum(dim=1)        # (B,)

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:
            return loss_per_sample


# ---------- Importance Reweighting ----------
class IRLoss(nn.Module):
    """
    Importance reweighting loss
    """

    def __init__(self, M, loss_code: str, reduction: str = "mean", eps: float = 1e-28):
        super().__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.M = torch.as_tensor(M, dtype=torch.float32)
        self.loss_code = loss_code
        self.reduction = reduction
        self.eps = eps

    # def forward(self, logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

    #     z = z.long()

    #     # 1. Compute probabilities and log-probabilities
    #     logp = self.logsoftmax(logits)     # (B, C) = log p
    #     p = logp.exp()                     # (B, C) = p

    #     # 2. Compute Q as a constant (no gradient flow)
    #     # Using a context manager makes the 'constant' intent very clear
    #     with torch.no_grad():
    #         M = self.M.to(logits.device)
    #         Mp = p @ M.T
            
    #         # Add a small epsilon to avoid division by zero
    #         Q = p / (Mp + 1e-9)

    #     if self.loss_code == "cross_entropy":
    #         # 1. Gather the probability for the true class z
    #         pz = p.gather(1, z.view(-1, 1)).squeeze(1)
    #         pz = pz.clamp_min(self.eps)
            
    #         # 2. Gather the weight Q for the true class z (matching shapes)
    #         Qz = Q.gather(1, z.view(-1, 1)).squeeze(1)
            
    #         # Now both Qz and torch.log(pz) are shape (B,)
    #         loss_per_sample = -Qz * torch.log(pz)
            
    #     else:
    #         # For other scoring rules, compute the full matrix S first
    #         # Note: I used 'p' here as 'r' was likely a typo in your snippet
    #         S = Q * scoring_matrix(p, self.loss_code)
            
    #         # Then gather the specific loss for target class z
    #         loss_per_sample = S.gather(1, z.view(-1, 1)).squeeze(1)

    #     if self.reduction == "mean":
    #         return loss_per_sample.mean()
    #     elif self.reduction == "sum":
    #         return loss_per_sample.sum()
    #     return loss_per_sample

    def forward(self, logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = z.long().view(-1)   # z 是 weak-label index, shape [B]

    # 1) p(y|x)
        logp = self.logsoftmax(logits)   # [B, C]
        p = logp.exp()                   # [B, C]
        B, C = p.shape

    # 2) 统一把 M 变成 [C, K]
        M = self.M.to(logits.device).float()
        if M.shape[0] == C:
            M_cz = M          # [C, K]
        elif M.shape[1] == C:
            M_cz = M.T        # [C, K]
        else:
            raise ValueError(f"Incompatible shapes: p={p.shape}, M={M.shape}")

    # 3) 计算 posterior 权重 Q[b,c] = p(y=c|x,z)
        with torch.no_grad():
        # weak_probs[b,k] = P(z=k | x)
            weak_probs = (p @ M_cz).clamp_min(self.eps)   # [B, K]

        # P(observed z_i | x_i)
            pz_weak = weak_probs.gather(1, z.unsqueeze(1)).squeeze(1).clamp_min(self.eps)  # [B]

        # M_obs[b,c] = P(z_i | y=c)
            M_obs = M_cz[:, z].T.contiguous().clamp_min(self.eps)   # [B, C]

        # posterior weights over classes
            Q = (p * M_obs) / pz_weak.unsqueeze(1)   # [B, C]
            Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(self.eps)

    # 4) 用 Q 对“按类别的loss”加权（不要再用 z 去 gather p 了）
        if self.loss_code == "cross_entropy":
        # classwise CE = -log p_c
            loss_per_sample = -(Q * logp).sum(dim=1)

        else:
        # 你原来的 scoring_matrix(p, ...) 如果返回 [B,C]，这行就可以
            S = scoring_matrix(p, self.loss_code)   # [B, C]
            loss_per_sample = (Q * S).sum(dim=1)

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample



class PiCOLoss(nn.Module):
    """
    """

    def __init__(self, loss_code: str, reduction: str = "mean",
                 eps: float = 1e-28):

        super().__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.loss_code = loss_code
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, z: torch.Tensor,
                Q: torch.Tensor) -> torch.Tensor:
        """
        Commputes the forward loss for pseudo-targets Q.
        
        Parameters
        ----------
        logits : torch.Tensor
            The model outputs (before softmax)
        Q : torch.Tensor
            The pseudo-targets.
        """

        # p 和你原来的一样：log_softmax + exp
        logp = self.logsoftmax(logits)     # (B, C) = log p
        p = logp.exp()                     # (B, C) = p

        # ---------- E-step: 责任 Q（数值上 posterior，反向里当常数） ----------
        Q = Q.detach()                     # 非常重要：Q 不反传梯度

        if self.loss_code == "cross_entropy":
            # ---------- 原始 MC 的部分：L_MC = E_Q[-log p] ----------
            # scoring_matrix(p, "cross_entropy") = -log(p)，
            # 这里直接用 logp 更干净：-log p = -logp.exp().log() = -logp
            S = -logp                      # (B, C)
            loss_per_sample = (Q * S).sum(dim=1)   # (B,)

        else:
            # ---------- 其它 scoring rule: 保持原始 MC 定义 ----------
            S = scoring_matrix(p, self.loss_code)       # (B, C)
            loss_per_sample = (Q * S).sum(dim=1)        # (B,)

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:
            return loss_per_sample


# ---------- Forward (plug-in) marginal-chain objective ----------
class ForwardProperLoss(nn.Module):
    def __init__(self, F_mat, loss_code: str, reduction: str = "mean",
                 eps: float = 1e-28):
        """
        Contains methods for training a model with a forward loss
        phi(z, p) = - z'·phi(F·p) , where phi is the proper loss specified by
        loss_code, z is the weak label and p is the model probabilistic
        prediction, and F is the forward matrix that maps p to the
        pseudo-target r = F p.

        Parameters
        F_mat : array-like
            The forward matrix F that maps model predictions to pseudo-targets.
        loss_code : str
            The code specifying which proper loss to use (e.g., "cross_entropy",
            "brier", "spherical", etc.).
        reduction : str, optional
            Specifies the reduction to apply to the output: "mean", "sum", or
            "none" (default is "mean").
        eps : float, optional
            A small value to avoid numerical issues (default is 1e-28).
        """

        super().__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.F = torch.as_tensor(F_mat, dtype=torch.float32)
        self.loss_code = loss_code
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward loss phi(z, p) = - z'·phi(F·p) , where phi is
        the proper loss specified by loss_code, z is the weak label, p is the
        model's predicted distribution, and F is the forward matrix that maps p
        to the pseudo-target r = F p.

        Parameters
        logits : torch.Tensor
            The model outputs (before softmax), shape (B, C).
        z : torch.Tensor
            The weak labels, shape (B,). Each entry is an integer class index.
        """

        z = z.long()
        logp = self.logsoftmax(logits)
        p = logp.exp()                     # (B, C)

        F = self.F.to(logits.device)       # (C, C)
        r = torch.matmul(F, p.T).T         # (B, C)

        # Cross entropy is treated separately to avoid log(0) issues
        if self.loss_code == "cross_entropy":
            # Take the r corresponding to the weak label z for each sample
            rz = r.gather(1, z.view(-1, 1)).squeeze(1)
            # Avoid log(0) by clamping to a minimum value
            rz = rz.clamp_min(self.eps)
            # Now compute the loss
            loss_per_sample = -torch.log(rz)
        else:
            S = scoring_matrix(r, self.loss_code)
            loss_per_sample = S.gather(1, z.view(-1, 1)).squeeze(1)

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:
            return loss_per_sample


class UpperBoundWeakProperLoss(nn.Module):
    """
    真正把 weak-label 目标做成：E_{y~posterior}[ proper_loss(f, y) ]（分离型、对 true classes）
    M: (D, C)  p(z | y)
    logits: (B, C)
    z_weak: (B,)
    """
    def __init__(self, M, loss_code: str, reduction="mean", eps=1e-12):
        super().__init__()
        self.register_buffer("M", torch.as_tensor(M, dtype=torch.float32))
        self.loss_code = loss_code
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, z_weak: torch.Tensor) -> torch.Tensor:
        z_weak = z_weak.long()
        logp = F.log_softmax(logits, dim=1)         # (B,C)
        f = logp.exp()                               # (B,C)

        M = self.M.to(device=logits.device, dtype=logits.dtype)  # (D,C)
        Mz = M[z_weak]                                # (B,C)  取每个样本对应的那一行

        # posterior r over true classes: r_j ∝ m_{zj} f_j
        num = Mz * f                                  # (B,C)
        den = num.sum(dim=1, keepdim=True) + self.eps # (B,1)
        r = (num / den).detach()                      # (B,C)  MM: stop-grad on r

        # ---------- losses on f with soft target r ----------
        if self.loss_code == "cross_entropy":
            loss = -(r * logp).sum(dim=1)             # (B,)

        elif self.loss_code in ("brier", "squared", "mse"):
            # E_y ||f - e_y||^2 = ||f - r||^2 + const(r)，优化时用 ||f-r||^2 就行
            loss = ((f - r) ** 2).sum(dim=1)

        elif self.loss_code.startswith("ps_"):
            beta = float(self.loss_code.split("_", 1)[1])
            if beta <= 1.0:
                raise ValueError("ps_β 要求 beta > 1")

            f_safe = f + self.eps
            num = (r * f_safe.pow(beta)).sum(dim=1)   # (B,)
            denom = f_safe.pow(beta).sum(dim=1).pow(1.0 / beta) + self.eps
            loss = - num / (beta * denom)             # (B,)

        elif self.loss_code.startswith("tsallis_"):
            alpha = float(self.loss_code.split("_", 1)[1])
            if abs(alpha - 1.0) < 1e-6:
                loss = -(r * logp).sum(dim=1)
            else:
                a = alpha - 1.0
                f_safe = f + self.eps
                # 用 log 形式更稳：f^{alpha-1} = exp((alpha-1) log f)
                f_a1 = torch.exp((alpha - 1.0) * torch.log(f_safe))
                term = (1.0 - alpha * f_a1) / a        # (B,C)
                main = (r * term).sum(dim=1)           # E_y[phi(f_y)]
                extra = f_safe.pow(alpha).sum(dim=1)   # (B,)
                loss = main + extra
        else:
            raise ValueError(f"Unknown loss_code={self.loss_code}")

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

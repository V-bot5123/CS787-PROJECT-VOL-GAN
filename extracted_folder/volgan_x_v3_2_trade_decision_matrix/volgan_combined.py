import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import fftpack, interpolate, optimize
from scipy.stats import norm
LOT_SIZE = 25
RISK_FREE_RATE = 0.065
TICK_FLOOR = 0.05
V_MIN = 10
OI_MIN = 100
BROKERAGE_PER_ORDER = 20.0
EXCH_TXN_RATE = 0.00050
STT_RATE_SELL = 0.00100
SEBI_FEE_RATE = 0.000001
STAMP_RATE_BUY = 0.00003
GST_RATE = 0.18
def half_spread_proxy(premium: float) -> float:
    return max(0.50, 0.015 * premium)
def hedge_cost(lots: int, lot_size: int = LOT_SIZE) -> float:
    return 0.05 * lots * lot_size
TENORS_DAYS = np.array([7, 14, 21, 30, 45, 60, 90, 180], dtype=np.float64)
TENORS_TAU = TENORS_DAYS / 365.0
NUM_TENORS = len(TENORS_DAYS)
NUM_STRIKES = 40
STD_MONEYNESS_X = np.linspace(-3.0, 3.0, NUM_STRIKES)
ATM_REF_VOL = 0.15
FFT_N = 4096
FFT_ETA = 0.25
DAMPING_ALPHA_DEFAULT = 1.5
MARTINGALE_TOLERANCE = 1e-8
HUBER_DELTA = 1.0
TEMPORAL_ANCHOR_LAMBDA = 0.05
MAX_CALIBRATION_RMSE = 0.03
MAX_NEWTON_ITER = 15
VEGA_CUTOFF = 1e-5
BRENT_BOUNDS = (0.001, 4.0)
THETA_Y = 0.70
TAU_HEALTHY = 2.50
MIN_EDGE_INR = 250.0
KAPPA = 3.0
RISK_BUDGET_INR = 25000.0
def black76_price(F: float, K: float, tau: float, sigma: float,
                  r: float, option_type: str = 'CE') -> float:
    if tau <= 0.0 or sigma <= 0.0:
        return max(0.0, (F - K) * math.exp(-r * tau)) if option_type == 'CE' else \
               max(0.0, (K - F) * math.exp(-r * tau))
    vol_sqrt_tau = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * tau) / vol_sqrt_tau
    d2 = d1 - vol_sqrt_tau
    df = math.exp(-r * tau)
    if option_type == 'CE':
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
def black76_vega(F: float, K: float, tau: float,
                 sigma: float, r: float) -> float:
    if tau <= 0.0 or sigma <= 0.0:
        return 0.0
    vol_sqrt_tau = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * tau) / vol_sqrt_tau
    return F * math.exp(-r * tau) * math.sqrt(tau) * norm.pdf(d1)
def invert_iv_black76(price: float, F: float, K: float, tau: float,
                     r: float, option_type: str = 'CE') -> float:
    df = math.exp(-r * tau)
    intrinsic = max(0.0, df * (F - K)) if option_type == 'CE' else \
                max(0.0, df * (K - F))
    if price <= intrinsic + TICK_FLOOR:
        return np.nan
    sigma = 0.20
    for _ in range(MAX_NEWTON_ITER):
        p = black76_price(F, K, tau, sigma, r, option_type)
        diff = p - price
        if abs(diff) < 1e-5:
            return sigma
        v = black76_vega(F, K, tau, sigma, r)
        if abs(v) < VEGA_CUTOFF:
            break
        sigma -= diff / v
        if sigma <= BRENT_BOUNDS[0] or sigma >= BRENT_BOUNDS[1]:
            break
    def objective(s):
        return black76_price(F, K, tau, s, r, option_type) - price
    try:
        return optimize.brentq(objective, *BRENT_BOUNDS, xtol=1e-5)
    except (ValueError, RuntimeError):
        return np.nan
def vg_characteristic_function(u, F: float, tau: float,
                               sigma: float, nu: float, theta: float):
    arg = 1.0 - theta * nu - 0.5 * sigma * sigma * nu
    if arg <= 0.0:
        return np.full_like(u, np.nan, dtype=np.complex128)
    omega = (1.0 / nu) * math.log(arg)
    term1 = np.exp(1j * u * (math.log(F) + omega * tau))
    term2 = (1.0 - 1j * u * theta * nu + 0.5 * sigma * sigma * nu * (u ** 2)) ** (-tau / nu)
    return term1 * term2
def verify_vg_martingale(F: float, tau: float,
                         sigma: float, nu: float, theta: float) -> bool:
    phi_neg_i = vg_characteristic_function(-1j, F, tau, sigma, nu, theta)
    if np.isnan(phi_neg_i):
        return False
    return abs(phi_neg_i.real / F - 1.0) < MARTINGALE_TOLERANCE
def carrmadan_fft_call_prices(F: float, tau: float, r: float,
                              sigma: float, nu: float, theta: float,
                              strikes_eval: np.ndarray) -> np.ndarray:
    alpha = DAMPING_ALPHA_DEFAULT
    for _ in range(5):
        cond = 1.0 - (alpha + 1.0) * theta * nu - 0.5 * sigma ** 2 * nu * (alpha + 1.0) ** 2
        if cond > 0.01:
            break
        alpha *= 0.7
    if cond <= 0.01:
        raise ValueError("No admissible Carr-Madan damping parameter for VG inputs")
    N = FFT_N
    eta = FFT_ETA
    lambda_spacing = 2.0 * math.pi / (N * eta)
    b = 0.5 * N * lambda_spacing
    v = np.arange(N) * eta
    u_eval = v - (alpha + 1.0) * 1j
    phi_vals = vg_characteristic_function(u_eval, F, tau, sigma, nu, theta)
    denom = (alpha ** 2 + alpha - v ** 2) + 1j * (2.0 * alpha + 1.0) * v
    psi = math.exp(-r * tau) * phi_vals / denom
    weights = np.empty(N, dtype=np.float64)
    weights[0] = eta / 3.0
    for j in range(1, N):
        weights[j] = (eta / 3.0) * (4.0 if j % 2 == 1 else 2.0)
    weights[-1] = eta / 3.0
    fft_input = np.exp(1j * b * v) * psi * weights
    fft_res = fftpack.fft(fft_input)
    k_grid = -b + lambda_spacing * np.arange(N)
    call_grid = (np.exp(-alpha * k_grid) / math.pi) * fft_res.real
    target_k = np.log(strikes_eval)
    cs = interpolate.CubicSpline(k_grid, call_grid, extrapolate=False)
    return np.maximum(0.0, np.nan_to_num(cs(target_k), nan=0.0))
def calibrate_vg_tenor(F: float, tau: float, r: float,
                       market_ivs: np.ndarray, mask: np.ndarray,
                       strikes: np.ndarray,
                       prior_theta: np.ndarray = None) -> tuple[np.ndarray, bool]:
    valid_idx = np.where((mask == 1) & ~np.isnan(market_ivs))[0]
    if len(valid_idx) < 8:
        return np.array([0.15, 0.20, -0.10]), False
    target_ivs = market_ivs[valid_idx]
    target_strikes = strikes[valid_idx]
    if prior_theta is None:
        prior_theta = np.array([np.median(target_ivs), 0.20, -0.10])
    def huber_objective(p):
        sig, nu, th = p
        if sig < 0.05 or sig > 1.5 or nu < 0.01 or nu > 1.5 or \
           (th * nu + 0.5 * sig * sig * nu) >= 0.95:
            return 1e6
        try:
            model_calls = carrmadan_fft_call_prices(F, tau, r, sig, nu, th,
                                                   target_strikes)
            model_ivs = []
            for cp, K in zip(model_calls, target_strikes):
                iv = invert_iv_black76(cp, F, K, tau, r, 'CE')
                model_ivs.append(iv if not np.isnan(iv) else sig)
            model_ivs = np.array(model_ivs)
            resid = target_ivs - model_ivs
            huber = np.where(np.abs(resid) <= HUBER_DELTA,
                             0.5 * resid ** 2,
                             HUBER_DELTA * (np.abs(resid) - 0.5 * HUBER_DELTA))
            temporal_pen = TEMPORAL_ANCHOR_LAMBDA * np.sum((p - prior_theta) ** 2)
            return float(np.mean(huber) + temporal_pen)
        except Exception:
            return 1e6
    res = optimize.minimize(huber_objective, prior_theta,
                            method='Nelder-Mead',
                            options={'maxiter': 150, 'xatol': 1e-3, 'fatol': 1e-4})
    sig_opt, nu_opt, th_opt = res.x
    martingale_ok = verify_vg_martingale(F, tau, sig_opt, nu_opt, th_opt)
    try:
        calls = carrmadan_fft_call_prices(F, tau, r, sig_opt, nu_opt, th_opt,
                                          target_strikes)
        fitted_ivs = np.array([invert_iv_black76(c, F, K, tau, r, 'CE')
                               for c, K in zip(calls, target_strikes)])
        fitted_ivs = np.nan_to_num(fitted_ivs, nan=sig_opt)
        rmse = float(np.sqrt(np.mean((target_ivs - fitted_ivs) ** 2)))
    except Exception:
        rmse = 1.0
    qc_passed = martingale_ok and (rmse <= MAX_CALIBRATION_RMSE)
    return np.array([sig_opt, nu_opt, th_opt]), qc_passed
def get_market_surface(trading_date: str = '2024-01-15',
                       mock: bool = True,
                       regime: str = 'normal') -> tuple[np.ndarray, np.ndarray]:
    X = np.full((NUM_TENORS, NUM_STRIKES), np.nan, dtype=np.float64)
    M = np.zeros((NUM_TENORS, NUM_STRIKES), dtype=np.int8)
    if not mock:
        raise NotImplementedError("Live market data are not implemented; use mock=True")
    if mock:
        date_hash = sum(ord(c) for c in trading_date)
        rng = np.random.RandomState(date_hash)
        base_atm = 0.14
        skew = 0.08
        smile = 0.15
        if regime == 'calm':
            base_atm = 0.11 + rng.uniform(-0.01, 0.01)
            skew = 0.05
        elif regime == 'turbulent':
            base_atm = 0.22 + rng.uniform(-0.02, 0.03)
            skew = 0.12
        elif regime == 'crisis':
            base_atm = 0.34 + rng.uniform(-0.03, 0.05)
            skew = 0.18
            smile = 0.25
        for i, tau in enumerate(TENORS_TAU):
            atm_vol = base_atm + 0.03 * math.sqrt(tau) + rng.uniform(-0.005, 0.005)
            s_i = ATM_REF_VOL * math.sqrt(tau)
            for j, x_std in enumerate(STD_MONEYNESS_X):
                k = x_std * s_i
                is_liquid = abs(x_std) <= (2.0 if tau <= 0.15 else 1.2)
                if is_liquid:
                    iv = atm_vol - skew * k + smile * (k ** 2) + rng.uniform(-0.004, 0.004)
                    X[i, j] = max(0.04, iv)
                    M[i, j] = 1
        for i in range(NUM_TENORS):
            valid = np.where(M[i] == 1)[0]
            if len(valid) > 0:
                X[i, :valid[0]] = X[i, valid[0]]
                X[i, valid[-1]:] = X[i, valid[-1]]
                nan_idx = np.where(np.isnan(X[i]))[0]
                if len(nan_idx):
                    X[i, nan_idx] = np.interp(nan_idx, valid, X[i, valid])
            else:
                X[i, :] = 0.15
    return X, M
def get_residual_tensor(trading_date: str = '2024-01-15',
                       mock: bool = True,
                       regime: str = 'normal') -> dict:
    X, M = get_market_surface(trading_date, mock=mock, regime=regime)
    S0 = 24500.0
    r = RISK_FREE_RATE
    F_vec = np.zeros(NUM_TENORS, dtype=np.float64)
    sigma_model = np.zeros_like(X)
    vega_grid = np.zeros_like(X)
    row_qc = np.zeros(NUM_TENORS, dtype=bool)
    for i, tau in enumerate(TENORS_TAU):
        F_i = S0 * math.exp(r * tau)
        F_vec[i] = F_i
        strikes = F_i * np.exp(STD_MONEYNESS_X * ATM_REF_VOL * math.sqrt(tau))
        p_opt, qc_ok = calibrate_vg_tenor(F_i, tau, r, X[i], M[i], strikes)
        row_qc[i] = qc_ok
        calls_model = carrmadan_fft_call_prices(
            F_i, tau, r, p_opt[0], p_opt[1], p_opt[2], strikes
        )
        for j, (K, call_p) in enumerate(zip(strikes, calls_model)):
            iv_m = invert_iv_black76(call_p, F_i, K, tau, r, 'CE')
            sigma_model[i, j] = iv_m if not np.isnan(iv_m) else p_opt[0]
            vega_grid[i, j] = black76_vega(F_i, K, tau, sigma_model[i, j], r)
    R = X - sigma_model
    R[M == 0] = 0.0
    mad_baseline = 0.015
    R_tilde = R / mad_baseline
    return {
        'X': X,
        'M': M,
        'R': R,
        'R_tilde': R_tilde,
        'vega': vega_grid,
        'F': F_vec,
        'row_qc': row_qc
    }
def init_filter_bank(conv_layer: nn.Conv2d, r_channel_idx: int = 1):
    with torch.no_grad():
        nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out', nonlinearity='leaky_relu')
        if conv_layer.bias is not None:
            nn.init.zeros_(conv_layer.bias)
        lap_4conn = torch.tensor([[0., -1., 0.],
                                  [-1., 4., -1.],
                                  [0., -1., 0.]])
        lap_8conn = torch.tensor([[-1., -1., -1.],
                                  [-1., 8., -1.],
                                  [-1., -1., -1.]])
        lap_strike = torch.tensor([[0., 0., 0.],
                                   [1., -2., 1.],
                                   [0., 0., 0.]])
        lap_expiry = torch.tensor([[0., 1., 0.],
                                   [0., -2., 0.],
                                   [0., 1., 0.]])
        base_kernels = [lap_4conn, lap_8conn, lap_strike, lap_expiry]
        for i in range(8):
            base_k = base_kernels[i % 4]
            noise = 0.05 * torch.randn_like(base_k) if i >= 4 else 0.0
            k_final = base_k + noise
            conv_layer.weight[i, :, :, :] = 0.0
            conv_layer.weight[i, r_channel_idx, :, :] = k_final
class CellPersistenceNet(nn.Module):
    def __init__(self, C: int = 5, F_dim: int = 32, d: int = 64, a: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(C, F_dim, kernel_size=3, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(F_dim, F_dim, kernel_size=3, padding=1, padding_mode='replicate')
        self.act = nn.LeakyReLU(0.1)
        self.lstm = nn.LSTM(F_dim, d, batch_first=True)
        self.Wa = nn.Linear(d, a)
        self.v = nn.Linear(a, 1, bias=False)
        self.head = nn.Linear(d, 1)
        init_filter_bank(self.conv1)
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, C, H, W = x.shape
        x_reshaped = x.reshape(B * L, C, H, W)
        f = self.act(self.conv1(x_reshaped))
        f = self.act(self.conv2(f))
        f = f.reshape(B, L, -1, H, W).permute(0, 3, 4, 1, 2)\
                .reshape(B * H * W, L, -1)
        h, _ = self.lstm(f)
        att_score = self.v(torch.tanh(self.Wa(h))).squeeze(-1)
        alpha = torch.softmax(att_score, dim=-1)
        ctx = (alpha.unsqueeze(-1) * h).sum(dim=1)
        logits = self.head(ctx).reshape(B, H, W)
        alpha = alpha.reshape(B, H, W, L)
        return logits, alpha
def predict_persistence(model: CellPersistenceNet, x_seq: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits, alpha = model(x_seq)
        y_hat = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        alpha_np = alpha.squeeze(0).cpu().numpy()
    return y_hat, alpha_np
class VolatilityVAE(nn.Module):
    def __init__(self, input_dim: int = 320, latent_dim: int = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1)
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)
        self.dec_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, input_dim)
        )
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_net(x)
        return self.fc_mu(h), self.fc_log_var(h)
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_net(z)
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
def vae_loss_function(x: torch.Tensor, x_recon: torch.Tensor,
                      mu: torch.Tensor, log_var: torch.Tensor,
                      mask: torch.Tensor,
                      beta: float = 1.0,
                      lambda_free: float = 0.05,
                      imputed_weight: float = 0.3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = torch.where(mask > 0.5, 1.0, imputed_weight)
    recon_loss = ((x - x_recon) ** 2 * weights).sum() / weights.sum()
    kl = -0.5 * (1.0 + log_var - mu ** 2 - torch.exp(log_var))
    kl = torch.clamp(kl, min=lambda_free).sum(dim=-1).mean()
    total = recon_loss + beta * kl
    return total, recon_loss, kl
@torch.no_grad()
def encode_mu(model: VolatilityVAE, X: np.ndarray,
              train_mean: np.ndarray, train_std: np.ndarray) -> np.ndarray:
    model.eval()
    x_std = (X - train_mean) / (train_std + 1e-6)
    x_tensor = torch.from_numpy(x_std.reshape(1, -1)).float()
    mu, _ = model.encode(x_tensor)
    return mu.squeeze(0).cpu().numpy()
@torch.no_grad()
def get_anomaly_score(model: VolatilityVAE, X: np.ndarray, M: np.ndarray,
                      train_mean: np.ndarray, train_std: np.ndarray,
                      median_train: float = 0.05, mad_train: float = 0.02) -> float:
    model.eval()
    x_std = (X - train_mean) / (train_std + 1e-6)
    x_tensor = torch.from_numpy(x_std.reshape(1, -1)).float()
    mu, _ = model.encode(x_tensor)
    recon = model.decode(mu).squeeze(0).cpu().numpy().reshape(NUM_TENORS, NUM_STRIKES)
    real_idx = np.where(M == 1)
    if len(real_idx[0]) == 0:
        return 999.0
    err = np.mean((x_std[real_idx] - recon[real_idx]) ** 2)
    return float((err - median_train) / (mad_train + 1e-6))
def build_condition_vector(mu: np.ndarray,
                           rv5: float = 0.14,
                           rv21: float = 0.15,
                           last_return: float = 0.002) -> np.ndarray:
    c = np.concatenate([mu, [rv5, rv21, last_return]])
    return c.astype(np.float32)
class VolGANGenerator(nn.Module):
    def __init__(self, z_dim: int = 32, c_dim: int = 8, out_dim: int = 321):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + c_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, out_dim)
        )
    def forward(self, z: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([z, c], dim=-1)
        raw = self.net(inp)
        dlogS = 0.25 * torch.tanh(raw[:, 0])
        dlogIV = 0.50 * torch.tanh(raw[:, 1:]).reshape(-1, 8, 40)
        return dlogS, dlogIV
class VolGANCritic(nn.Module):
    def __init__(self, in_dim: int = 321, c_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + c_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, delta: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([delta, c], dim=-1)
        return self.net(inp)
def compute_arbitrage_penalties(X_surfaces: torch.Tensor,
                               tau_vec: np.ndarray = TENORS_TAU) -> tuple[torch.Tensor, torch.Tensor]:
    N, H, W = X_surfaces.shape
    tau_t = torch.tensor(tau_vec, dtype=X_surfaces.dtype, device=X_surfaces.device).reshape(1, H, 1)
    W_tot = (X_surfaces ** 2) * tau_t
    calendar_viol = W_tot[:, :-1, :] - W_tot[:, 1:, :]
    L_cal = torch.relu(calendar_viol).mean()
    dk_values = np.diff(STD_MONEYNESS_X)[0] * ATM_REF_VOL * np.sqrt(tau_vec)
    dk = torch.as_tensor(dk_values, dtype=X_surfaces.dtype,
                         device=X_surfaces.device).reshape(1, H, 1)
    k_values = STD_MONEYNESS_X[None, :] * ATM_REF_VOL * np.sqrt(tau_vec)[:, None]
    k_mid = torch.as_tensor(k_values[:, 1:-1], dtype=X_surfaces.dtype,
                            device=X_surfaces.device).reshape(1, H, W - 2)
    w_mid = W_tot[:, :, 1:-1].clamp(min=1e-8)
    w_plus = W_tot[:, :, 2:]
    w_minus = W_tot[:, :, :-2]
    w_prime = (w_plus - w_minus) / (2.0 * dk)
    w_second = (w_plus - 2.0 * w_mid + w_minus) / (dk ** 2)
    first_term = (1.0 - k_mid * w_prime / (2.0 * w_mid)) ** 2
    term2 = (w_prime ** 2 / 4.0) * (1.0 / w_mid + 0.25)
    g_k = first_term - term2 + 0.5 * w_second
    L_fly = torch.relu(-g_k).mean()
    return L_cal, L_fly
def scenario_pnl(legs: list[dict], X_curr: np.ndarray,
                 dlogS: np.ndarray, dlogIV: np.ndarray,
                 h_days: int = 1) -> np.ndarray:
    N = len(dlogS)
    pnl = np.zeros(N, dtype=np.float64)
    r = RISK_FREE_RATE
    X_new = X_curr[None, :, :] * np.exp(dlogIV)
    for leg in legs:
        K = leg['K']
        F0 = leg['F0']
        tau0 = leg['tau']
        qty = leg['qty']
        lot = leg['lot']
        v0 = leg['v0']
        op_type = leg['type']
        tenor_idx = int(leg['tenor_idx'])
        strike_idx = int(leg['strike_idx'])
        tau1 = max(1e-5, tau0 - h_days / 365.0)
        F1 = F0 * np.exp(dlogS)
        for n in range(N):
            iv1 = float(X_new[n, tenor_idx, strike_idx])
            v1 = black76_price(F1[n], K, tau1, iv1, r, op_type)
            pnl[n] += qty * lot * (v1 - v0)
    exit_friction = BROKERAGE_PER_ORDER * 2.0
    return pnl - exit_friction
def calculate_conservative_cvar(pnls_gan: np.ndarray,
                               pnls_crisis: np.ndarray = None,
                               pnls_hist: np.ndarray = None) -> float:
    def single_cvar(arr):
        if arr is None or len(arr) == 0:
            return 0.0
        sorted_arr = np.sort(arr)
        tail = max(1, int(0.05 * len(sorted_arr)))
        return float(np.mean(sorted_arr[:tail]))
    c1 = single_cvar(pnls_gan)
    c2 = single_cvar(pnls_crisis) if pnls_crisis is not None else c1
    c3 = single_cvar(pnls_hist) if pnls_hist is not None else c1
    return min(c1, c2, c3)
@torch.no_grad()
def simulate_tail_risk(generator: VolGANGenerator, c_t: np.ndarray,
                       X_curr: np.ndarray, legs: list[dict],
                       num_scenarios: int = 1000) -> float:
    generator.eval()
    c_tensor = torch.from_numpy(c_t).float().unsqueeze(0).repeat(num_scenarios, 1)
    z_noise = torch.randn(num_scenarios, 32)
    dlogS_t, dlogIV_t = generator(z_noise, c_tensor)
    dlogS = dlogS_t.cpu().numpy()
    dlogIV = dlogIV_t.cpu().numpy()
    pnls = scenario_pnl(legs, X_curr, dlogS, dlogIV, h_days=1)
    return calculate_conservative_cvar(pnls)
def compute_net_edge_inr(R: float, vega_unit: float, rho_hat: float,
                         lots: int, lot_size: int,
                         premium: float, side: str = 'sell') -> float:
    q = lots * lot_size
    gross = rho_hat * vega_unit * abs(R) * q
    spread = 2.0 * half_spread_proxy(premium) * q
    turnover = premium * q
    costs = spread
    costs += BROKERAGE_PER_ORDER * 2.0
    costs += EXCH_TXN_RATE * turnover * 2.0
    if side == 'sell':
        costs += STT_RATE_SELL * turnover
    costs += SEBI_FEE_RATE * turnover * 2.0
    if side == 'buy':
        costs += STAMP_RATE_BUY * turnover
    taxable = (BROKERAGE_PER_ORDER * 2.0 +
               (EXCH_TXN_RATE + SEBI_FEE_RATE) * turnover * 2.0)
    costs += GST_RATE * taxable
    costs += hedge_cost(lots, lot_size)
    return float(gross - costs)
def python_risk_gate(state: dict) -> dict:
    def abort(code, msg):
        state['decision'] = 'ABORT'
        state['abort_reason'] = f"{code}: {msg}"
        return state
    need = ['resid_iv', 'vega_unit', 'y_hat', 'anomaly_z',
            'cvar_95', 'net_edge', 'row_qc_ok']
    for k in need:
        v = state.get(k)
        if v is None or (isinstance(v, (int, float)) and not math.isfinite(v)):
            return abort('G0_MISSING', f"Field '{k}' missing or non‑finite.")
    state['gate_results']['G0_MISSING'] = True
    if not state['row_qc_ok']:
        return abort('G0_QC', "Row calibration QC failed.")
    state['gate_results']['G0_QC'] = True
    if not state.get('models_validated', False):
        return abort('G0_MODEL', 'Untrained or unvalidated ML models; demo cannot authorize trades.')
    state['gate_results']['G0_MODEL'] = True
    return abort('G0_LIVE_DISABLED', 'Trade authorization disabled for all research runs.')

def llm_audit_node(state: dict) -> dict:
    state['rationale'] = (
        f"Trade disabled ({state['abort_reason']}). "
        f"Residual IV={state['resid_iv']:.4f}, "
        f"y_hat={state['y_hat']:.2f}, "
        f"anomaly z={state['anomaly_z']:.2f}, "
        f"CVaR95={state['cvar_95']:.2f}."
    )
    return state

def run_trade_decision_agent(contract: dict, cell: tuple,
                             resid_iv: float, vega_unit: float,
                             y_hat: float, anomaly_z: float,
                             cvar_95: float, row_qc_ok: bool) -> dict:
    net_edge = compute_net_edge_inr(R=resid_iv,
                                    vega_unit=vega_unit,
                                    rho_hat=0.50,
                                    lots=contract.get('lots', 1),
                                    lot_size=contract.get('lot_size', LOT_SIZE),
                                    premium=contract.get('premium', 150.0),
                                    side=contract.get('side', 'sell'))
    state = {
        'contract': contract,
        'cell': cell,
        'resid_iv': resid_iv,
        'vega_unit': vega_unit,
        'y_hat': y_hat,
        'anomaly_z': anomaly_z,
        'cvar_95': cvar_95,
        'net_edge': net_edge,
        'row_qc_ok': row_qc_ok,
        'models_validated': bool(contract.get('models_validated', False)),
        'decision': None,
        'abort_reason': None,
        'gate_results': {},
        'rationale': None
    }
    state = python_risk_gate(state)
    state = llm_audit_node(state)
    return state
def main(date: str = '2024-01-15',
         window_days: int = 10,
         regime: str = 'normal'):
    print("\n=== VolGAN-X PIPELINE (SYNTHETIC, UNTRAINED DEMO) ===")
    print(f"Date: {date} | Look‑back: {window_days} days | Regime: {regime}\n")
    torch.manual_seed(sum(ord(c) for c in date) + window_days)
    data = get_residual_tensor(date, mock=True, regime=regime)
    X_curr = data['X']
    M_mask = data['M']
    R_grid = data['R']
    vega_grid = data['vega']
    F_vec = data['F']
    row_qc = data['row_qc']
    print(f"Surface X shape: {X_curr.shape}, real cells: {M_mask.sum()}/{X_curr.size}")
    print(f"Residual max mispricing: {np.abs(R_grid).max():.4f}")
    vae = VolatilityVAE()
    mu = encode_mu(vae, X_curr, np.full_like(X_curr, 0.15), np.full_like(X_curr, 0.03))
    anomaly_z = get_anomaly_score(vae, X_curr, M_mask,
                                  np.full_like(X_curr, 0.15), np.full_like(X_curr, 0.03))
    c_t = build_condition_vector(mu, rv5=0.14, rv21=0.15, last_return=0.002)
    print(f"Latent μ: {np.round(mu, 4)}")
    print(f"Anomaly z‑score: {anomaly_z:.2f}")
    pers_model = CellPersistenceNet()
    seq = torch.randn(1, window_days, 5, 8, 40)
    y_hat_surface, _ = predict_persistence(pers_model, seq)
    print(f"Persistence surface shape: {y_hat_surface.shape}, mean: {y_hat_surface.mean():.3f}")
    row_qc_2d = np.repeat(row_qc[:, None], NUM_STRIKES, axis=1)
    valid_mask = (M_mask == 1) & (R_grid > 0.015) & row_qc_2d
    if np.any(valid_mask):
        idx = np.unravel_index(np.argmax(np.where(valid_mask, R_grid, -np.inf)), R_grid.shape)
    else:
        idx = (int(np.argmax(row_qc)), NUM_STRIKES // 2)
    i, j = map(int, idx)
    cand_R = float(R_grid[i, j])
    cand_vega = float(vega_grid[i, j])
    cand_F = float(F_vec[i])
    cand_tau = float(TENORS_TAU[i])
    K = cand_F * math.exp(STD_MONEYNESS_X[j] * ATM_REF_VOL * math.sqrt(cand_tau))
    cand_yhat = float(y_hat_surface[i, j])
    gen = VolGANGenerator()
    entry_premium = black76_price(cand_F, K, cand_tau, float(X_curr[i, j]),
                                  RISK_FREE_RATE, 'CE')
    leg = [{'type': 'CE', 'K': K, 'F0': cand_F,
            'tau': cand_tau, 'qty': -1, 'lot': LOT_SIZE,
            'v0': entry_premium, 'tenor_idx': i, 'strike_idx': j}]
    cvar_95 = simulate_tail_risk(gen, c_t, X_curr, leg, num_scenarios=2000)
    contract = {
        'strike': K,
        'expiry': f"{int(cand_tau*365)}d",
        'type': 'CE',
        'S0': 24500.0,
        'F': cand_F,
        'lots': 1,
        'lot_size': LOT_SIZE,
        'premium': entry_premium,
        'side': 'sell' if cand_R > 0 else 'buy',
        'models_validated': False
    }
    final_state = run_trade_decision_agent(contract, (i, j),
                                           resid_iv=cand_R,
                                           vega_unit=cand_vega,
                                           y_hat=cand_yhat,
                                           anomaly_z=anomaly_z,
                                           cvar_95=cvar_95,
                                           row_qc_ok=bool(row_qc[i] and valid_mask[i, j]))
    print("\n=== FINAL TRADE TICKET ===")
    print(f"Selected cell: ({i}, {j})  Strike: {K:.1f}  Tenor: {int(cand_tau*365)} days")
    print(f"Decision: {final_state['decision']}")
    print(f"Abort reason: {final_state['abort_reason']}")
    print(f"Net edge (INR per lot): {final_state['net_edge']:.2f}")
    print(f"Gates passed: {final_state['gate_results']}")
    print("Audit memo:")
    print(final_state['rationale'])
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Research-only VolGAN-X demo or historical CSV audit")
    parser.add_argument('--mode', choices=['demo', 'historical', 'train', 'matrix'], default='demo',
                        help='Synthetic untrained demo, or historical CSV research-only audit')
    parser.add_argument('--data-dir', default=None,
                        help='Folder containing nifty_options_YYYY-MM-DD.csv; default: auto-find bundled data/')
    parser.add_argument('--output-dir', default='.', help='Directory for JSON and CSV research reports')
    parser.add_argument('--date', default='2024-01-15', help='Trading date (YYYY‑MM‑DD)')
    parser.add_argument('--window', type=int, default=10, help='Look‑back window in days')
    parser.add_argument('--regime', default='normal',
                        choices=['calm', 'normal', 'turbulent', 'crisis'],
                        help='Synthetic market regime')
    parser.add_argument('--vae-epochs', type=int, default=12)
    parser.add_argument('--persistence-epochs', type=int, default=12)
    parser.add_argument('--gan-steps', type=int, default=16)
    parser.add_argument('--top-n', type=int, default=5, help='Top ranked research pairs per date to export')
    parser.add_argument('--explain-date', default=None, help='Explain G0-G7 for the top-ranked pair on a date, YYYY-MM-DD')
    args = parser.parse_args()
    if args.mode == 'train':
        from volgan_model_research import run_model_research
        report = run_model_research(args.data_dir, args.output_dir,
                                    args.vae_epochs, args.persistence_epochs, args.gan_steps)
        print(f"Real-data model fit: {report['data']['dates']} dates / {report['data']['observed_iv_cells']} observed IV cells")
        print(f"Supervised windows (train/validation/test): {report['chronology']['train_windows']}/{report['chronology']['validation_windows']}/{report['chronology']['test_windows']}")
        print('VAE standardized-IV NMSE (train/validation/test):',
              *(report['vae'][k+'_metrics']['nmse'] for k in ('train','validation','test')))
        print('VAE reconstruction MSE (train/validation/test):',
              *(report['vae'][k+'_metrics']['mse'] for k in ('train','validation','test')))
        print('Persistence unweighted BCE (train/validation/test):',
              *(report['persistence']['train'][k+'_metrics']['bce'] for k in ('train','val','test')))
        print('Persistence Brier (train/validation/test):',
              *(report['persistence']['train'][k+'_metrics']['brier'] for k in ('train','val','test')))
        print('Per-epoch logs: vae_epoch_metrics.csv, persistence_epoch_metrics.csv; GAN steps: gan_step_metrics.csv')
        print('All scores exploratory; validation/test sets have only two windows each.')
        print(f"GAN status: {report['wgan_gp']['status']}")
        from trade_decision_matrix import run_decision_matrix, print_decision_matrix
        matrix_report = run_decision_matrix(args.data_dir, args.output_dir, args.top_n)
        print_decision_matrix(args.output_dir, explain_date=args.explain_date)
        print('BLOCKED_LIVE: Model training remains experimental; no executable trade authorization.')
    elif args.mode == 'matrix':
        from trade_decision_matrix import run_decision_matrix, print_decision_matrix
        report = run_decision_matrix(args.data_dir, args.output_dir, args.top_n)
        print(f"NIFTY options candidate pairs: {report['candidate_pairs']} over {report['dates']} dates")
        print_decision_matrix(args.output_dir, explain_date=args.explain_date)
    elif args.mode == 'historical':
        from historical_threshold_research import run_historical_research
        report = run_historical_research(args.data_dir, args.output_dir)
        print(f"Historical source: {report['data']['source_files']} CSVs, {report['data']['source_rows']} rows")
        print(f"Chronological train/holdout: {report['training_period']['dates']}/{report['holdout_period']['dates']} dates")
        print(f"Preliminary parity-proxy MIN_EDGE: {report['provisional_research_min_edge_inr']}")
        from trade_decision_matrix import run_decision_matrix, print_decision_matrix
        run_decision_matrix(args.data_dir, args.output_dir, args.top_n)
        print_decision_matrix(args.output_dir, explain_date=args.explain_date)
        print('BLOCKED_LIVE: Historical analysis is research-only, not an execution approval.')
    else:
        main(date=args.date, window_days=args.window, regime=args.regime)

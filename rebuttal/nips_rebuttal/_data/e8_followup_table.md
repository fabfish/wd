# E8 follow-up peaks

| tag | wd_sched | peak_acc | peak_λ0 | Δ vs fixed | n |
|---|---|---:|---:|---:|---:|
| joint_T100 | fixed | 66.67 | 0.0001 | +0.00 | 5 |
| joint_T100 | cosine | 76.42 | 0.001 | +9.75 | 5 |
| joint_T100 | linear | 76.17 | 0.001 | +9.50 | 5 |
| joint_T100 | step | 75.36 | 0.0005 | +8.69 | 5 |
| joint_T100 | cosine_restarts | 75.11 | 0.001 | +8.44 | 5 |
| joint_T200 | fixed | 62.39 | 0.0005 | +0.00 | 3 |
| joint_T200 | step | 76.31 | 0.0005 | +13.92 | 3 |
| joint_T200 | cosine_restarts | 76.88 | 0.001 | +14.49 | 3 |

## vs E4 constant λ (same η₀, T, SGDM)

E8 follow-up: unified η₀=0.1, T=100, SGDM. Joint = same m(t) on η and λ.

| method | best_acc | λ₀ / note |
|---|---:|---|
| cosine LR + fixed λ (E4-ours C/∑η) | 76.72 | 0.0005982 |
| cosine LR + fixed λ (default 5e-4) | 76.73 | 0.0005 |
| cosine LR + fixed λ (oracle over λ₀ grid) | 77.28 | λ₀=0.001 |
| joint m(t) = fixed (peak over λ₀) | 66.67 | λ₀=0.0001 |
| joint m(t) = cosine (peak over λ₀) | 76.42 | λ₀=0.001 |
| joint m(t) = linear (peak over λ₀) | 76.17 | λ₀=0.001 |
| joint m(t) = step (peak over λ₀) | 75.36 | λ₀=0.0005 |
| joint m(t) = cosine_restarts (peak over λ₀) | 75.11 | λ₀=0.001 |

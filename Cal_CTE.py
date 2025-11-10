import pandas as pd
import numpy as np

b = 0.286e-9
eta = 1.25
Gm = 26.32e9
nu = 0.33
rho_m = 2.70e3
T_test = 298
T_process = 773
Delta_T = T_process - T_test
particle_properties = {
    "SiC": {"Delta_alpha": 1.96e-6, "rho_p": 3.21e3},
    "TiC": {"Delta_alpha": 1.62e-6, "rho_p": 4.90e3},
    "ZrB2": {"Delta_alpha": 1.69e-6, "rho_p": 5.80e3},
    "TiB2": {"Delta_alpha": 1.56e-6, "rho_p": 4.52e3},
    "TiCN": {"Delta_alpha": 1.76e-6, "rho_p": 5.08e3},
    "Al3Ti": {"Delta_alpha": 1.10e-6, "rho_p": 3.36e3}
}


def calculate_strengthening(excel_file, d_p_column='d_p'):

    data = pd.read_excel(excel_file)
    data[d_p_column] = data[d_p_column] * 1e-9  # nm -> m


    particle_columns = [col for col in data.columns if col != d_p_column]
    print("Particle type", particle_columns)

    results = {}

    for particle in particle_columns:
        w_p = data[particle].copy()
        w_p = w_p.apply(lambda x: x/100 if x > 1 else x)
        w_m = 1 - w_p

        # Volume fraction Calculation
        rho_p = particle_properties[particle]["rho_p"]
        V_p = (w_p / rho_p) / ((w_p / rho_p) + (w_m / rho_m))
        V_p = V_p.clip(0, 0.99)

        # CTE
        Delta_alpha = particle_properties[particle]["Delta_alpha"]
        rho_CTE = (12 * Delta_alpha * Delta_T * V_p) / (b * data[d_p_column] * (1 - V_p))
        rho_CTE = rho_CTE.clip(0)
        Delta_sigma_CTE = eta * Gm * b * np.sqrt(rho_CTE) / 1e6  # Pa -> MPa

        # Orowan
        lambda_ = data[d_p_column] * np.sqrt(np.maximum(np.pi / (6 * V_p) - 2/3, 0.0))
        Delta_sigma_Or = np.where(lambda_>0,
                                  (1.2 * Gm * b * np.log(0.8165 * data[d_p_column] / b)) /
                                  (np.pi * lambda_ * np.sqrt(1 - nu)) / 1e6,
                                  0.0)

        df_result = pd.DataFrame({
            'w_p': w_p,
            'V_p': V_p,
            'Delta_sigma_CTE(MPa)': Delta_sigma_CTE,
            'Delta_sigma_Or(MPa)': Delta_sigma_Or,
            'd_p(m)': data[d_p_column]
        })

        results[particle] = df_result

    return results

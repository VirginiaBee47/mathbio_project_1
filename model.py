# 3-compartment ODE model (numbers): Juveniles (J), Adult Females (F), Adult Males (M)
# Recruitment: Beverton-Holt driven by females: R(F) = alpha * F / (1 + beta * F)
# Continuous-time ODEs with instantaneous fishing mortality applied to adults.
#
# This script:
# - defines the ODE system
# - integrates it with scipy.integrate.solve_ivp
# - plots time series of J, F, M and catch (instantaneous rate)
# - provides a function `run_scenario` so you can try different fishing mortalities and selectivities
#
# Time units: years. States are numbers (abundance).
# To change parameters or initial conditions, modify the `params` or `y0` below and re-run.

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from data import NATURAL_MORTALITY, FISHING_MORTALITY

# ------------------------- Model definition -------------------------
# Definitions:
# Landings:         Live fish brought back to shore
# Dead-releases:    Dead fish released back to the water
# Releases:         All fish released back to the water
# Total Catch:      All fish caught
# Total Harvest:    All fish caught + not able to reproduce => landings + dead releases

# To calculate instantaneous fishing mortality for adults, use the exploitation rate equation
def fishing_mortality(year):
    H = FISHING_MORTALITY[year - 1986][0]
    C = FISHING_MORTALITY[year - 1986][1]
    return -np.log(1-H/C)

def bh_recruitment(F, alpha, beta):
    """Beverton-Holt recruitment driven by female abundance F."""
    return alpha * F / (1.0 + beta * F)

def odes(t, y, params):
    """ODE system: y = [J, F, M]"""
    J, F, M = y
    # unpack params
    m = params['m']           # maturation rate (1/years)
    mu_J = params['mu_J']     # juvenile natural mortality
    mu_F = params['mu_F']     # female natural mortality
    mu_M = params['mu_M']     # male natural mortality
    p = params['p']           # proportion of recruits that become female
    alpha = params['alpha']   # BH alpha
    beta = params['beta']     # BH beta
    F_fish = params['F_fish'] # instantaneous fishing mortality applied to adults
    s_F = params['s_F']       # female selectivity (0-1)
    s_M = params['s_M']       # male selectivity (0-1)
    
    # recruitment depends only on females (female-driven BH)
    R = bh_recruitment(F, alpha, beta)
    
    # ODEs
    dJdt = R - (m + mu_J) * J
    dFdt = p * m * J - (mu_F + s_F * F_fish) * F
    dMdt = (1.0 - p) * m * J - (mu_M + s_M * F_fish) * M
    
    return [dJdt, dFdt, dMdt]

def catch_rate(F, M, F_fish, s_F, s_M):
    """Instantaneous catch rate (numbers per year) from adults at time t."""
    return s_F * F_fish * F + s_M * F_fish * M

def natural_mortality(population, params):
    # Considering that juvenile maturation is a parameter, the
    # classification must be based on that parameter
    # NATURAL_MORTALITY[0][0][1] = 0.924 = [FEMALE][AGE0][MORTALITY]
    if population == 'J':
        selected_year_min = 0 
        selected_year_max = int(np.ceil(1/params['m']))
    elif population == 'F' or population == 'M':
        selected_year_min = int(np.ceil(1/params['m']))
        selected_year_max = 13
    else:
        raise ValueError("Population must be 'J', 'F', or 'M'")
    
    print(selected_year_min, selected_year_max)
    if population == 'J':
        mu = np.mean(NATURAL_MORTALITY[0:2, selected_year_min:selected_year_max, 1])
    elif population == 'F':
        mu = np.mean(NATURAL_MORTALITY[0,   selected_year_min:selected_year_max, 1])
    elif population == 'M':
        mu = np.mean(NATURAL_MORTALITY[1,   selected_year_min:selected_year_max, 1])

    return mu

# Spawning potential ratio
def maturity(age, m):
    """Proportion mature at age given maturation rate m (1/years)."""
    k=1
    return 1 / (1 + np.exp(-k*(age - 1 / m)))

def survivorship(age, fish_mort, nat_mort):
    return np.exp(-(fish_mort + nat_mort) * age)

def length_to_weight(length):
    pass


# How to actually get the data:
# Table 16 (in 1000s)
# Abundance = # of mature fish
# Age-0 recruits = # of juveniles
# Convert Age-0 recruits to Age-0 biomass
# Subtract that from total biomass
# multiply by SSB/Total Biomass to find Female population
# F = Abundance * 1000 * SSB / (Total Biomass - Age-0 Biomass)
# M = Abundance * 1000 * (1 - SSB / (Total Biomass - Age-0 Biomass))

# ------------------------- Default parameters & initial state -------------------------
if __name__ == "__main__":
    params = {
        'm': 1/1.0,       # juveniles mature on avg in 1 years
        'mu_J': 0.5,      # juvenile natural mortality (1/yr)
        'mu_F': 0.2,      # female natural mortality (1/yr)
        'mu_M': 0.2,      # male natural mortality (1/yr)
        'p': 0.5,         # sex ratio at maturation: 50% female
        'alpha': 800.0,   # BH alpha (recruits per female at low density) -- tuneable
        'beta': 0.002,    # BH beta (density-dependence) -- tuneable
        'F_fish': 0.25,   # instantaneous fishing mortality (1/yr) applied to adults
        's_F': 1.0,       # female selectivity (1 = fully vulnerable)
        's_M': 1.0,       # male selectivity
    }

    # initial abundances (numbers)
    y0 = [2000.0, 1000.0, 1000.0]  # J, F, M at t=0 (year 0)

    t_span = (0.0, 30.0)  # simulate 30 years
    t_eval = np.linspace(t_span[0], t_span[1], 301)  # output times

    # ------------------------- Run baseline scenario -------------------------
    sol = solve_ivp(fun=lambda t, y: odes(t, y, params),
                    t_span=t_span, y0=y0, t_eval=t_eval, method='RK45', rtol=1e-6)

    J = sol.y[0]
    F = sol.y[1]
    M = sol.y[2]
    time = sol.t
    catch_inst = catch_rate(F, M, params['F_fish'], params['s_F'], params['s_M'])

    # ------------------------- Plot results -------------------------
    plt.figure(figsize=(8,5))
    plt.plot(time, J, label='Juveniles (J)')
    plt.plot(time, F, label='Adult females (F)')
    plt.plot(time, M, label='Adult males (M)')
    plt.xlabel('Time (years)')
    plt.ylabel('Abundance (numbers)')
    plt.title('3-compartment ODE model (Beverton-Holt recruitment)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(time, catch_inst, label='Instantaneous catch rate (numbers/yr)')
    plt.xlabel('Time (years)')
    plt.ylabel('Catch (numbers per year)')
    plt.title('Instantaneous catch rate over time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # ------------------------- Utility: run scenarios -------------------------
    def run_scenario(params, y0, t_span=(0,30), t_eval=None):
        """Run the model for a given params dict and initial state; return a dict with results."""
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 301)
        sol = solve_ivp(fun=lambda t, y: odes(t, y, params),
                        t_span=t_span, y0=y0, t_eval=t_eval, method='RK45', rtol=1e-6)
        J = sol.y[0]; F = sol.y[1]; M = sol.y[2]
        catch_inst = catch_rate(F, M, params['F_fish'], params['s_F'], params['s_M'])
        return {'t': sol.t, 'J': J, 'F': F, 'M': M, 'catch': catch_inst, 'params': params}

    # Example: try a sex-biased scenario (male-targeted fishery)
    params_male_targeted = params.copy()
    params_male_targeted['s_F'] = 0.2
    params_male_targeted['s_M'] = 1.0
    params_male_targeted['F_fish'] = 0.35

    res_mt = run_scenario(params_male_targeted, y0, t_span=t_span, t_eval=t_eval)

    plt.figure(figsize=(8,5))
    plt.plot(res_mt['t'], res_mt['J'], label='Juveniles (J)')
    plt.plot(res_mt['t'], res_mt['F'], label='Adult females (F)')
    plt.plot(res_mt['t'], res_mt['M'], label='Adult males (M)')
    plt.xlabel('Time (years)')
    plt.ylabel('Abundance (numbers)')
    plt.title('Male-targeted fishery: s_F=0.2, s_M=1.0, F_fish=0.35')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(res_mt['t'], res_mt['catch'], label='Instantaneous catch rate (numbers/yr)')
    plt.xlabel('Time (years)')
    plt.ylabel('Catch (numbers per year)')
    plt.title('Catch in male-targeted scenario')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print final-year summary for baseline and male-targeted scenario
    def summary_at_end(res):
        t_final = res['t'][-1]
        return {
            'year': t_final,
            'J_final': float(res['J'][-1]),
            'F_final': float(res['F'][-1]),
            'M_final': float(res['M'][-1]),
            'catch_final': float(res['catch'][-1])
        }

    print('Baseline final-year summary:', summary_at_end({'t': time, 'J': J, 'F': F, 'M': M, 'catch': catch_inst}))
    print('Male-targeted final-year summary:', summary_at_end(res_mt))

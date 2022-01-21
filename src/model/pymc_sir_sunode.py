import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.figure_factory as ff
import scipy.stats
from scipy.integrate import odeint
import pymc3 as pm
import arviz as az
import sunode
import sunode.wrappers.as_theano
# sunode object to customise solver configs
lib = sunode._cvodes.lib
import warnings
warnings.filterwarnings('ignore')

class SIR_model_sunode():
 
    def __init__(self, covid_data) :
        # ------------------------- Covid_data object -----------------------#
        self.covid_data = covid_data
        # ------------------------- Setup SIR model, but has to be called explicitly to run ------------------------#
        self._setup_SIR_model_data()
 

    def _setup_SIR_model_data(self):
        
        self.time_range = np.arange(0,len(self.covid_data.cases_obs),1)
        self.I0 = self.covid_data.cases_obs[0]
        self.S0 = self.covid_data.N - self.I0
        self.S_init = self.S0 / self.covid_data.N
        self.I_init = self.I0 / self.covid_data.N
        self.cases_obs_scaled = self.covid_data.cases_obs / self.covid_data.N
 

    def SIR_sunode_rhs_ode(self, t, y, p):
        
        rhs_ode_dict = {
            'S': -p.lam * y.S * y.I,
            'I': p.lam * y.S * y.I - p.mu * y.I,
            #'R': p.f * p.mu * y.I ## f in considered as 1 for this exercice
        }
        
        return rhs_ode_dict
    
    
    def build_pymc_model(self, likelihood, prior):
        # ------------------------- Metadata --------------------------------#
        self.likelihood = likelihood
        self.prior = prior
 
        with pm.Model() as model:
            ## pymc RVs - Priors
            sigma = pm.HalfCauchy('sigma', self.likelihood['sigma'], shape=1)
            lam = pm.Lognormal('lambda', self.prior['lam'], self.prior['lambda_std']) # 1.5, 1.5
            mu = pm.Lognormal('mu', self.prior['mu'], self.prior['mu_std'])           # 1.5, 1.5
            ## sunode ODE equation and gradients computed using sunode
            res, _, problem, solver, _, _ = sunode.wrappers.as_theano.solve_ivp(
                y0={'S': (self.S_init, ()), 'I': (self.I_init, ()),},
                params={'lam': (lam, ()), 'mu': (mu, ()), '_dummy': (np.array(1.), ())},
                rhs=self.SIR_sunode_rhs_ode,
                tvals=self.time_range,
                t0=self.time_range[0]
            )
            ## raw sundials functions to customise sunode solver options
            ## powered by pysundials https://github.com/jmuhlich/pysundials/tree/master/doc
            lib.CVodeSStolerances(solver._ode, 1e-10, 1e-10)
            lib.CVodeSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
            lib.CVodeQuadSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
            lib.CVodeSetMaxNumSteps(solver._ode, 5000)
            lib.CVodeSetMaxNumStepsB(solver._ode, solver._odeB, 5000)
            ## pymc RVs - likelihood
            if(likelihood['distribution'] == 'lognormal'):
                I = pm.Lognormal('I', mu=res['I'], sigma=sigma, observed=self.cases_obs_scaled)
            elif(likelihood['distribution'] == 'normal'):
                I = pm.Normal('I', mu=res['I'], sigma=sigma, observed=self.cases_obs_scaled)
            elif(likelihood['distribution'] == 'students-t'):
                I = pm.StudentT( "I",  nu=likelihood['nu'],       # likelihood distribution of the data
                        mu=res['I'],     # likelihood distribution mean, these are the predictions from SIR
                        sigma=sigma,
                        observed=self.cases_obs_scaled
                        )
            R0 = pm.Deterministic('R0',lam/mu)
        
        self.pymc_model = model
        
        
    def plot_pymc_model_dag(self):
        
        dag_fig = pm.model_to_graphviz(self.pymc_model)
        return dag_fig
    
    def sample_posterior_pymc_model(self, n_samples, n_tune, n_chains, n_cores):
        
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.n_cores = n_cores
        
        try:
            self.pymc_model is not None
        except NotImplementedError as error:
            print('pymc3 model instance not found')

        with self.pymc_model:
            
            trace = pm.sample(self.n_samples, tune=self.n_tune, 
                              chains=self.n_chains, cores=self.n_cores)
            
        self.pymc_model_trace = trace
        
    
    def pymc_model_posterior_summary(self):
        
        trace_summary = az.summary(self.pymc_model_trace)
        return trace_summary
    
    
    def pymc_model_plot_posterior(self):
        
        data = az.from_pymc3(trace=self.pymc_model_trace)
        az.plot_posterior(data, round_to=2, point_estimate='mode', hdi_prob=0.95)

    def pymc_model_plot_traces(self):
        
        axes = az.plot_trace(self.pymc_model_trace)
        axes.ravel()[0].figure
    
    
    def pymc_model_plot_interactive_trace(self, trace='R0'):
        
        fig = ff.create_distplot([self.pymc_model_trace[trace]], bin_size=0.5, group_labels=['x'])
        # Add title
        fig.update_layout(title_text='Curve and Rug Plot')
        fig.update_xaxes(range=[0,7])
        return fig
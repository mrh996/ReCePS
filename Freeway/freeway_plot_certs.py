
import scipy.special
from scipy.stats import norm, binom_test,sem
from statsmodels.stats.proportion import proportion_confint
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

def get_cvar_cert_time_t(estimate, t, eps,sigma):
	erf = scipy.special.erf(math.sqrt(t+1) * eps/(2*math.sqrt(2)*sigma))
	cvar = 1. if estimate > erf else estimate/erf
	return cvar * erf

def get_exact_time_t(estimate, t, eps,sigma):
	return norm.cdf(norm.ppf(estimate) - math.sqrt(t+1) * eps/(sigma))

def _lower_confidence_bound(NA: int, N: int, alpha: float) -> float:
    return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
def get_exact_total(estimate, eps,sigma):
	return norm.cdf(norm.ppf(estimate) -  eps/(sigma))

def compute_one_minus_cdf_function(sorted_values, lb_func):
	last_value = 0
	total = 0
	for idx,v in enumerate(sorted_values):
		if (v == last_value):
			continue
		else:
			one_minus_cdf = (len(sorted_values) - idx)/float(len(sorted_values)) # 1-cdf for the interval between x_(i-1) and x_i
			lb_one_minus_cdf = lb_func(one_minus_cdf)
			total += lb_one_minus_cdf * (v-last_value) 
			last_value = v

	print(total)
	return total
def dkw_cohen(one_minus_cdf_empirical, n, alpha, eps,sigma):
	dkw = max(one_minus_cdf_empirical - math.sqrt(math.log(2/alpha)/(2*n)), 0)
	return get_exact_total(dkw,eps,sigma)

plt.figure(figsize=(8.4,4.8))

for sigma in [12.75,25.5]:
	data = torch.load('freeway_sigma_'+str(sigma)+'_1/best_model.zip_evals_10000.pth')
	sorted_values = list([sum(x)  for x in data])
	sorted_values.sort()
	sorted_values = torch.tensor(sorted_values,device='cuda')
	vals = []
	for eps in np.arange(0*255,0.41*255, 0.01*255):
		print(eps/255)
		vals.append(compute_one_minus_cdf_function(sorted_values, lambda x : dkw_cohen(x, 10000, 0.05,eps, sigma)).cpu())
	plt.plot(np.arange(0,0.41, 0.01),vals, color=('blue' if sigma ==12.75 else 'cornflowerblue'), linestyle=('-' if sigma ==12.75 else '--') ,label="σ = " + str(sigma/255) ) #"Policy Smoothing: Certified\nLower Bound (σ = " + str(sigma/255) +')')


# attack_mags_nonzero =[25.5, 51.0, 76.5,102.0]

# attack_vals =  [torch.tensor([sum(x) for x in torch.load('freeway_sigma_0.0_2/best_model.zip_evals_10000.pth')]).mean().item()]
# attack_sems = [sem(torch.tensor([sum(x) for x in torch.load('freeway_sigma_0.0_2/best_model.zip_evals_10000.pth')]))]
# for attack_mag in attack_mags_nonzero:
# 	attack_val = None
# 	attack_sem = None
# 	for i,thresh in enumerate([0.0,0.06,0.12]):
# 		cur_val = (torch.tensor( [sum(x) for x in torch.load('freeway_sigma_0.0_2/best_model.zip_evals_1000_attack_eps_'+str(attack_mag)+'_attack_step_count_multiplier_2_attack_step_2.5500000000000003_threshold_'+str(thresh)+'.pth')]).mean().item())
# 		if (attack_val is None or cur_val < attack_val):
# 			attack_val = cur_val
# 			attack_sem = (sem(torch.tensor( [sum(x) for x in  torch.load('freeway_sigma_0.0_2/best_model.zip_evals_1000_attack_eps_'+str(attack_mag)+'_attack_step_count_multiplier_2_attack_step_2.5500000000000003_threshold_'+str(thresh)+'.pth')])))
# 	attack_vals.append(attack_val)
# 	attack_sems.append(attack_sem)
# attack_mags = [0] + attack_mags_nonzero
# plt.errorbar([x/255 for x in attack_mags],attack_vals,  yerr= attack_sems, color='red',  linestyle ="--",label="Undefended"  )



# styles = ['-.','--',":"]
# attack_mags_nonzero = [25.5, 51.0, 76.5,102.0]
# for j,sigma in enumerate([12.75]):#,25.6]):
# 	attack_vals =  [torch.tensor([sum(x) for x in torch.load('freeway_sigma_'+str(sigma)+'_1/best_model.zip_evals_10000.pth')]).mean().item()]
# 	attack_sems = [sem(torch.tensor([sum(x) for x in  torch.load('freeway_sigma_'+str(sigma)+'_1/best_model.zip_evals_10000.pth')]))]
# 	for attack_mag in attack_mags_nonzero:
# 		attack_val = None
# 		attack_sem = None
# 		for i,thresh in enumerate([0.0,0.06,0.12]):
# 			cur_val = (torch.tensor( [sum(x) for x in torch.load('freeway_sigma_'+str(sigma)+'_1/best_model.zip_evals_1000_smooth_attack_eps_'+str(attack_mag)+'_attack_step_count_multiplier_2_attack_step_2.5500000000000003_threshold_'+str(thresh)+'_num_smoothing_points_128.pth')]).mean().item())
# 			if (attack_val is None or cur_val < attack_val):
# 				attack_val = cur_val
# 				attack_sem = (sem(torch.tensor( [sum(x) for x in torch.load('freeway_sigma_'+str(sigma)+'_1/best_model.zip_evals_1000_smooth_attack_eps_'+str(attack_mag)+'_attack_step_count_multiplier_2_attack_step_2.5500000000000003_threshold_'+str(thresh)+'_num_smoothing_points_128.pth')])))
# 		attack_vals.append(attack_val)
# 		attack_sems.append(attack_sem)
# 	attack_mags = [0] + attack_mags_nonzero
# 	plt.errorbar([x/255 for x in attack_mags],attack_vals,  yerr= attack_sems, color='blue',  linestyle ="-",label="Policy Smoothing (σ = " + str(sigma/255.) + ')')





plt.legend()
plt.xlim(0,0.4)
plt.ylim(0,5)
plt.title('(c) Freeway (250 Frames)', fontsize=18)
plt.xlabel('Perturbation Budget', fontsize=14)
plt.ylabel('Certified Average Score', fontsize=14)
plt.savefig('freeway_certs.png',dpi=400,bbox_inches='tight')

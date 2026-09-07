import numpy as np
import math

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
	result = math.log(sigma_q/sigma_p) + (sigma_p ** 2 + (mu_p - mu_q) ** 2)/(2 * sigma_q ** 2) - 1/2
	return result

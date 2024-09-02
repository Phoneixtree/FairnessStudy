import numpy as np

def coupon_collector_simulation(M, m, target_probability=0.995, max_trials=100000):
    def single_simulation(M, m, k):
        collected = set()
        for _ in range(k):
            draw = np.random.choice(M, m, replace=True)
            collected.update(draw)
            if len(collected) == M:
                return True
        return False

    k = 0
    while k < max_trials:
        k += 1
        successful_trials = sum(single_simulation(M, m, k) for _ in range(1000))
        probability = successful_trials / 1000.0
        if probability >= target_probability:
            return k
    
    return None  # If we exceed the max_trials and do not find a solution

# Example usage
M = 545  # Number of different balls
m = 10   # Number of balls drawn each time
target_probability = 0.995

min_draws = coupon_collector_simulation(M, m, target_probability)
print(f"Minimum number of draws required to achieve probability greater than {target_probability}: {min_draws}")
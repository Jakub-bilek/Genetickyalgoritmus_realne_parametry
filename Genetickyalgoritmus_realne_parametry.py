import struct
import random
import numpy as np
import matplotlib.pyplot as plt

# 1. TESTOVACÍ FUNKCE (10D) A JEJICH DOMÉNY
def sphere_function(x):
    return sum(x**2)

def rosenbrock_function(x):
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin_function(x):
    A = 10
    return A * len(x) + sum(x**2 - A * np.cos(2 * np.pi * x))

BENCHMARKS = {
    'Sphere': {'func': sphere_function, 'bounds': (-5.12, 5.12)},
    'Rosenbrock': {'func': rosenbrock_function, 'bounds': (-2.048, 2.048)},
    'Rastrigin': {'func': rastrigin_function, 'bounds': (-5.12, 5.12)}
}

# 2. BINÁRNÍ ENKÓDOVÁNÍ A DEKÓDOVÁNÍ (32 bitů na dimenzi)
def float32_to_bits(val):
    packed = struct.pack('>f', float(val))
    integ = struct.unpack('>I', packed)[0]
    return [int(b) for b in f"{integ:032b}"]

def bits_to_float32(bits, bounds):
    integ = int("".join(map(str, bits)), 2)
    packed = struct.pack('>I', integ)
    val = struct.unpack('>f', packed)[0]
    # Pokud vznikne NaN nebo Inf, vrátí se náhodná hodnota z domény (ne statická hranice)
    if np.isnan(val) or np.isinf(val):
        return float(random.uniform(bounds[0], bounds[1]))
    return float(np.clip(val, bounds[0], bounds[1]))

def float_to_fixed_bits(val):
    val_clamped = np.clip(val, -32768.0, 32767.9999)
    sign = 0 if val_clamped >= 0 else 1
    abs_val = abs(val_clamped)
    integ_part = int(abs_val) & 0x7FFF
    frac_part = int(round((abs_val - integ_part) * (2**16)))
    if frac_part >= (2**16):
        integ_part += 1
        frac_part = 0
    bits_str = f"{sign:01b}{integ_part:015b}{frac_part:016b}"
    return [int(b) for b in bits_str]

def fixed_bits_to_float(bits, bounds):
    bits_str = "".join(map(str, bits))
    sign = -1 if bits_str[0] == '1' else 1
    integ_part = int(bits_str[1:16], 2)
    frac_part = int(bits_str[16:], 2) / (2**16)
    val = sign * (integ_part + frac_part)
    return float(np.clip(val, bounds[0], bounds[1]))

def float_to_bcd_bits(val):
    val_clamped = np.clip(val, -999.9999, 999.9999)
    sign = 0 if val_clamped >= 0 else 1
    abs_val = abs(val_clamped)
    digits_str = f"{abs_val:08.4f}".replace('.', '')[:7]
    bits = [sign]
    for d in digits_str:
        bits.extend([int(b) for b in f"{int(d):04b}"])
    bits.extend([0, 0, 0])
    return bits

def bcd_bits_to_float(bits, bounds):
    sign = -1 if bits[0] == 1 else 1
    digits = []
    for i in range(1, 29, 4):
        val = bits[i]*8 + bits[i+1]*4 + bits[i+2]*2 + bits[i+3]
        digits.append(str(min(val, 9)))
    num_str = "".join(digits[:3]) + "." + "".join(digits[3:])
    try:
        val = sign * float(num_str)
    except ValueError:
        val = 0.0
    return float(np.clip(val, bounds[0], bounds[1]))

ENCODERS = {
    'IEEE-754': (float32_to_bits, bits_to_float32),
    'Fixed-Point': (float_to_fixed_bits, fixed_bits_to_float),
    'BCD': (float_to_bcd_bits, bcd_bits_to_float)
}

# 3. GENETICKÉ ALGORITMY
def binary_ga(fit_func, dim, bounds, enc_type, max_evals=10000, pop_size=30, p_mut=0.01):
    to_bits, to_float = ENCODERS[enc_type]
    
    pop_real = [np.random.uniform(bounds[0], bounds[1], dim) for _ in range(pop_size)]
    # Zploštění do souvislého bitového řetězce o délce dim * 32
    pop_bits = [[bit for v in ind for bit in to_bits(v)] for ind in pop_real]
    
    fitnesses = [fit_func(ind) for ind in pop_real]
    evals = pop_size
    best_fit = min(fitnesses)
    history = [best_fit] * evals

    total_bits = dim * 32

    while evals < max_evals:
        new_bits = []
        
        # Elitismus
        best_idx = np.argmin(fitnesses)
        new_bits.append(pop_bits[best_idx][:])
        
        while len(new_bits) < pop_size:
            # Turnajová selekce 2 rodičů
            i1, i2 = random.sample(range(pop_size), 2)
            p1 = pop_bits[i1 if fitnesses[i1] < fitnesses[i2] else i2]
            
            i3, i4 = random.sample(range(pop_size), 2)
            p2 = pop_bits[i3 if fitnesses[i3] < fitnesses[i4] else i4]
            
            # Jednobodové křížení na celém 320-bitovém řetězci
            cross_pt = random.randint(1, total_bits - 1)
            off1 = p1[:cross_pt] + p2[cross_pt:]
            off2 = p2[:cross_pt] + p1[cross_pt:]
            
            # Mutace bit po bitu
            off1 = [1 - b if random.random() < p_mut else b for b in off1]
            off2 = [1 - b if random.random() < p_mut else b for b in off2]
            
            new_bits.extend([off1, off2])
            
        pop_bits = new_bits[:pop_size]
        
        # Dekódování zpět po 32bitových bloucích
        pop_real = []
        for ind_bits in pop_bits:
            chrom = []
            for d in range(dim):
                gene_bits = ind_bits[d*32 : (d+1)*32]
                chrom.append(to_float(gene_bits, bounds))
            pop_real.append(chrom)
        
        new_fitnesses = []
        for ind in pop_real:
            if evals >= max_evals:
                break
            f = fit_func(np.array(ind))
            evals += 1
            best_fit = min(best_fit, f)
            history.append(best_fit)
            new_fitnesses.append(f)

        fitnesses = new_fitnesses
            
    return history[:max_evals], best_fit


def real_ga(fit_func, dim, bounds, mut_type='normal', sigma=0.1, p_mut=0.1, max_evals=10000, pop_size=30):
    pop = [np.random.uniform(bounds[0], bounds[1], dim) for _ in range(pop_size)]
    fitnesses = [fit_func(ind) for ind in pop]
    evals = pop_size
    
    best_fit = min(fitnesses)
    history = [best_fit] * evals

    while evals < max_evals:
        new_pop = []
        
        # Elitismus
        best_idx = np.argmin(fitnesses)
        new_pop.append(pop[best_idx].copy())
        
        while len(new_pop) < pop_size:
            i1, i2 = random.sample(range(pop_size), 2)
            p1 = pop[i1 if fitnesses[i1] < fitnesses[i2] else i2]
            
            i3, i4 = random.sample(range(pop_size), 2)
            p2 = pop[i3 if fitnesses[i3] < fitnesses[i4] else i4]
            
            cross_pt = random.randint(1, dim - 1)
            off1 = np.concatenate([p1[:cross_pt], p2[cross_pt:]])
            off2 = np.concatenate([p2[:cross_pt], p1[cross_pt:]])
            
            for off in (off1, off2):
                for d in range(dim):
                    if random.random() < p_mut:
                        if mut_type == 'normal':
                            off[d] += np.random.normal(0, sigma)
                        elif mut_type == 'uniform':
                            off[d] = np.random.uniform(bounds[0], bounds[1])
                        off[d] = np.clip(off[d], bounds[0], bounds[1])
            
            new_pop.extend([off1, off2])
            
        pop = new_pop[:pop_size]
        
        new_fitnesses = []
        for ind in pop:
            if evals >= max_evals:
                break
            f = fit_func(ind)
            evals += 1
            best_fit = min(best_fit, f)
            history.append(best_fit)
            new_fitnesses.append(f)

        fitnesses = new_fitnesses
            
    return history[:max_evals], best_fit

# 4. EXPERIMENTY A VYHODNOCENÍ
def run_experiments():
    dim = 10
    max_evals = 10000
    runs = 10
    pop_size = 30
    
    variants = [
        ('IEEE-754', 'binary', 'IEEE-754'),
        ('Fixed-Point', 'binary', 'Fixed-Point'),
        ('BCD', 'binary', 'BCD'),
        ('Normal sigma=0.05 p=0.1', 'real', ('normal', 0.05, 0.1)),
        ('Normal sigma=0.1 p=0.1', 'real', ('normal', 0.1, 0.1)),
        ('Normal sigma=0.5 p=0.1', 'real', ('normal', 0.5, 0.1)),
        ('Normal sigma=0.1 p=0.05', 'real', ('normal', 0.1, 0.05)),
        ('Normal sigma=0.1 p=0.2', 'real', ('normal', 0.1, 0.2)),
        ('Real + Abs. mut.', 'real', ('uniform', 0.1, 0.1))
    ]
    
    fig = plt.figure(figsize=(24, 6), constrained_layout=True)
    grid = fig.add_gridspec(1, 6, width_ratios=[4.5, 2] * 3)
    axes = [fig.add_subplot(grid[0, i]) for i in (0, 2, 4)]
    legend_axes = [fig.add_subplot(grid[0, i]) for i in (1, 3, 5)]
    
    result_rows = []

    for f_idx, (f_name, f_info) in enumerate(BENCHMARKS.items()):
        ax = axes[f_idx]
        legend_ax = legend_axes[f_idx]
        print(f"---> Spouštím {f_name} pro {dim}D (Limit: {max_evals} evaluací)", flush=True)
        
        for v_name, v_kind, v_param in variants:
            all_histories = []
            final_bests = []
            
            for r in range(runs):
                np.random.seed(r * 100 + f_idx * 10)
                random.seed(r * 100 + f_idx * 10)
                
                if v_kind == 'binary':
                    hist, best = binary_ga(
                        f_info['func'], dim, f_info['bounds'], enc_type=v_param,
                        max_evals=max_evals, pop_size=pop_size
                    )
                else:
                    mut_type, sigma, p_mut = v_param
                    hist, best = real_ga(
                        f_info['func'], dim, f_info['bounds'], mut_type=mut_type,
                        sigma=sigma, p_mut=p_mut, max_evals=max_evals, pop_size=pop_size
                    )
                
                all_histories.append(hist)
                final_bests.append(best)
                
            avg_hist = np.mean(all_histories, axis=0)
            best_result = float(np.min(final_bests))
            mean_result = float(np.mean(final_bests))
            std_result = float(np.std(final_bests))
            result_rows.append((f_name, v_name, best_result, mean_result, std_result))
            
            ax.plot(avg_hist, label=v_name, linewidth=1.5)
            
        ax.set_yscale('log')
        ax.set_title(f"Konvergence: {f_name} ({dim}D)")
        ax.set_xlabel("Počet evaluací")
        ax.set_ylabel("Fitness (Log Scale)")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        handles, labels = ax.get_legend_handles_labels()
        legend_ax.axis('off')
        legend_ax.legend(handles, labels, loc='center left', fontsize=8, frameon=False)

    print("Dokončeno", flush=True)
    print("=" * 112)
    print(f"{'Funkce':<12} | {'Varianta':<26} | {'Nejlepší':>12} | {'Průměr':>12} | {'Směr. odch.':>12}")
    print("=" * 112)
    for f_name, v_name, best_result, mean_result, std_result in result_rows:
        print(f"{f_name:<12} | {v_name:<26} | {best_result:12.4e} | {mean_result:12.4e} | {std_result:12.4e}")

    table_fig, table_ax = plt.subplots(figsize=(16, max(6, 0.38 * (len(result_rows) + 1))))
    table_ax.axis('off')
    table_ax.set_title(f'Souhrn výsledků: {runs} běhy pro každou variantu', pad=16)
    table_ax.table(
        cellText=[
            [f_name, v_name, f"{best:.4e}", f"{mean:.4e}", f"{std:.4e}"]
            for f_name, v_name, best, mean, std in result_rows
        ],
        colLabels=['Funkce', 'Varianta', 'Nejlepší běh', 'Průměr', 'Směr. odchylka'],
        cellLoc='center',
        loc='center',
        colWidths=[0.14, 0.38, 0.16, 0.16, 0.16],
    ).auto_set_font_size(False)
    table_ax.tables[0].set_fontsize(9)
    table_ax.tables[0].scale(1, 1.35)

    plt.tight_layout()
    print("Vykresluji", flush=True)
    plt.show()

if __name__ == "__main__":
    run_experiments()
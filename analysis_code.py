from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import yoda
import glob
import pandas as pd
import warnings
plt.rcParams["font.family"] = "serif"

files = glob.glob('../data/*.yoda')
file_names = [file_name[8:-5] for file_name in files]
data = {file_name : yoda.read(file_, asdict=False) for file_, file_name in zip(files, file_names)}
file_names.sort()

event_shapes = [hist.name() for hist in data["bottom"]][0:10]

SM_factors = {
    "bottom" : 1,
    "gluon" : 1,
    "light" : 1
}


def data_getter(event_shape, particle, factor=1, finalised=True):
    # Assert the form of input required.
    particle = particle.lower()
    assert particle in file_names, "Particle is invalid!"
    event_shape = event_shape.upper()
    event_shapes = [hist.name().upper() for hist in data[particle]]
    assert event_shape in event_shapes, "Event shape is invalid!"
    assert finalised == True or finalised == False, "Finalised must be True or False!"

    width = 0

    all_event_shapes = data[particle]
    event_path = "/EP_H_BOOSTED/{}".format(event_shape) if finalised == True else "/RAW/EP_H_BOOSTED/{}".format(
        event_shape)
    for event in all_event_shapes:
        if event.path().upper() == event_path:
            bins, heights = [], []
            for b in event:
                bins.append(b.xMin())
                heights.append(b.height())

    width = bins[1] - bins[0]

    return np.array(bins), np.array(heights) * factor, width


def total_calculator(event_shape, factors=SM_factors):
    particle_datas = {}
    for particle in file_names:
        factor = factors[particle]
        particle_bins, particle_vals, _ = data_getter(event_shape, particle, factor)
        particle_datas[particle.upper()] = zip(particle_bins, particle_vals)

    bins = zip(*particle_datas.values()[0])[0]
    vals = []
    for idx, _ in enumerate(bins):
        counter = 0
        for particle in file_names:
            counter += particle_datas[particle.upper()][idx][1]
        vals.append(counter)

    width = bins[1]

    return np.array(bins), np.array(vals), width


def histogrammer(event_shape, factors=SM_factors, finalised=True, scale="linear"):

    fig, ax = plt.subplots(figsize=(10, 6))
    bins, vals, width = total_calculator(event_shape, factors)

    ax.bar(x=bins, height=vals, align="edge", alpha=0.3, label="Total", width=width, color="grey")

    colours = {"bottom": "b", "gluon": "g", "light": "r"}
    for particle in file_names:  # for each particle:
        factor = factors[particle]
        particle_bins, particle_vals, width = data_getter(event_shape, particle, factor, finalised=finalised)  # Get data
        ax.bar(x=particle_bins, height=particle_vals, label=particle.capitalize(), alpha=0.5, width=width, align="edge", color=colours[particle])

    ax.set_yscale(scale.lower())
    ax.set_title("HISTOGRAM OF {}".format(event_shape.upper()), fontsize=28)
    ax.set_xlabel("{}".format(event_shape.upper().replace("_", " ")), fontsize=16)
    ax.set_ylabel("NUMBER OF EVENTS", fontsize=16)
    ax.set_xticks(np.linspace(bins[0], bins[-1] + width, 11))
    ax.grid()
    ax.legend()

    return ax


def ranger(lower, upper, width):
    num_bins = (upper - lower) / width
    ranges = []

    for A in np.linspace(lower, upper, num_bins + 1):
        B_start = np.round(A + width, 5)
        x = np.round(upper - B_start, 5)
        y = np.round(x / width, 5)

        for B in np.linspace(start=B_start, stop=upper, num=int(y + 1)):
            ranges.append((np.round(A, 5), np.round(B, 5)))

    return ranges


def ROC_calculator(event_shape, target, range_):
    # Range format is a tuple of (lower, upper)
    lower, upper = range_[0], range_[1]

    particle_datas = {}
    for particle in file_names:
        particle_bins, particle_vals, _ = data_getter(event_shape, particle, finalised=True)
        particle_datas[particle.upper()] = zip(particle_bins, particle_vals)

    TPs = 0
    FNs = 0
    FPs = 0
    TNs = 0

    for x, y in particle_datas.pop(target.upper()):
        if x > lower and x < upper:
            TPs += y
        else:
            FNs += y

    for data in particle_datas.values():
        for x, y in data:
            if x > lower and x < upper:
                FPs += y
            else:
                TNs += y

    # Calculate the TP rate and FP rate as they are the Y and X axes of a ROC curve
    TP_rate = TPs / (TPs + FNs)
    FP_rate = FPs / (FPs + TNs)

    return TP_rate, FP_rate


def optimum_range_finder(event_shape, target, ranges, plotting=False):
    rates = []
    for range_ in ranges:
        TPR, FPR = ROC_calculator(event_shape, target, range_=range_)
        rates.append(
            (range_[0], range_[1], TPR - FPR)
        )

    lower, upper, score = zip(*rates)
    best_score = max(score)
    best_range = (np.round(lower[score.index(best_score)], 5), np.round(upper[score.index(best_score)], 5))

    if plotting == True:

        plt.figure(figsize=(10, 8))
        plt.scatter(lower, upper, c=score, cmap="Greens", s=5)
        plt.scatter(best_range[0], best_range[1], color="r", s=50, label="Optimum range: {}".format(best_range))
        plt.xticks(np.linspace(0, 1, 11)); plt.yticks(np.linspace(0, 1, 11))
        plt.xlabel("A"); plt.ylabel("B", rotation=0, labelpad=10)
        plt.grid()
        plt.title("{} ranges trialled for {}".format(target.capitalize(), event_shape.replace("_", " ").lower()))
        plt.legend()
        plt.savefig("../results/grid.png")

    return best_range, np.round(best_score, 5)


ranges_scores = {
    "bottom": [],
    "gluon": [],
    "light": []
}
for p in file_names:
    for es in event_shapes:
        bins, vals, width = data_getter(es, p)
        opt = optimum_range_finder(es, p, ranger(bins[0], bins[-1]+width, width), plotting=False)
        ranges_scores[p].append(opt)

rangesdf = pd.DataFrame(data=ranges_scores)
rangesdf.insert(0, "observable", event_shapes)
rangesdf.columns = rangesdf.columns.str.capitalize()


def particles_in_range(event_shape, range_, factors=SM_factors):
    bins, vals, width = total_calculator(event_shape, factors)
    bins, vals = list(bins), list(vals)

    best_lower, best_upper = range_[0], range_[1] - width  # from 0.00 to 1.0 hence add width to upper
    idx_of_lower, idx_of_upper = bins.index(np.round(best_lower, 5)), bins.index(np.round(best_upper, 5))

    total = 0
    for val in vals[idx_of_lower: idx_of_upper + 1]:  # Val of last bin is given by index of last
        total += val  # bin which is 0.98 hence minus width.

    return total, np.sqrt(total)


def cs_err(event_shape, target, range_):
    # We need err more gluon events in the gluon range
    bins, vals, width = data_getter(event_shape, target)
    bins, vals = list(bins), list(vals)

    _, err = particles_in_range(event_shape, range_)

    total_gluons = 0

    for val in vals[bins.index(np.round(range_[0], 5)): bins.index(np.round(range_[1] - width, 5))]:
        total_gluons += val

    percent_increase = (err / total_gluons) * 100

    return percent_increase


def histogrammer_with_ranges(event_shape, target, factors=SM_factors, plotting=True):
    bins, vals, width = data_getter(event_shape, target)
    ranges = ranger(bins[0], bins[-1] + width, width)
    best_range, score = optimum_range_finder(event_shape, target, ranges)

    if plotting == True:
        ax = histogrammer(event_shape, factors=factors)
        ax.axvline(best_range[0], color="purple", label=str(best_range[0]))
        ax.axvline(best_range[1], color="purple", label=str(best_range[1]))
        ax.legend()
        plt.show()

    return cs_err(event_shape, target, best_range), score


res = {
    "bottom": [],
    "gluon": [],
    "light": []
}
for particle in file_names:
    for es in ["thrust", "Thrust_minor", "Count_of_neutral_particles_in_VFS", "Count_of_charged_particles_in_VFS"]:
        bins, _, width = data_getter(es, particle)
        opt_range, _ = optimum_range_finder(es, particle, ranges=ranger(bins[0], bins[1]+width, width))
        constraint, _ = cs_err(es, particle, range_ = opt_range)
        res[particle].append(constraint)

df = pd.DataFrame(data=res)
df.insert(0, "observable", ["Thrust", "Thrust Minor", "Neutral Particle Count", "Charged Particle Count"])
df.columns = df.columns.str.capitalize()


def SM(event_shape, target):
    bins, vals, width = total_calculator(event_shape, factors=SM_factors)
    bins, vals = list(bins), list(vals)
    best_range, _ = optimum_range_finder(event_shape, target, ranger(bins[0], bins[-1], width))  # Bins are from 0.00 to 0.98, we want range
    N_SM, N_err = particles_in_range(event_shape, range_=best_range, factors=SM_factors)

    return N_SM, N_err, best_range


def BSM(event_shape, BSM_factors, N_SM, N_err, range_):  # Target is the particle's range we use

    N_BSM, _ = particles_in_range(event_shape, range_=range_, factors=BSM_factors)
    SM_dev = None
    if abs(N_BSM - N_SM) > N_err:
        SM_dev = True
    else:
        SM_dev = False

    return SM_dev


def factors(g_range, l_range, points):

    ranges = []
    for A in np.linspace(g_range[0], g_range[1], points):
        for B in np.linspace(l_range[0], l_range[1], points):
            ranges.append((np.round(A, 5), np.round(B, 5)))

    return ranges


def main(event_shape, target, gFactorRange, lFactorRange, points):
    N_SM, N_err, range_ = SM(event_shape, target)
    g, l = zip(*factors(gFactorRange, lFactorRange, points))

    result = []
    for factor1 in list(set(g)):
        for factor2 in list(set(l)):
            BSM_factors = {"bottom": 1, "gluon": factor1, "light": factor2}
            res = BSM(event_shape, BSM_factors=BSM_factors, N_SM=N_SM, N_err=N_err, range_=range_)
            result.append(
                ((factor1, factor2), res)
            )

    return result


grange = [0.97, 1.03]
lrange = [0.84, 1.16]
resolution = 101

light = main(event_shape = "thrust_minor",
              target = "light",
              gFactorRange = grange,
              lFactorRange = lrange,
              points=resolution)

gluon = main(event_shape = "thrust_minor",
              target = "gluon",
              gFactorRange = grange,
              lFactorRange = lrange,
              points=resolution)


def part_calc(particle):
    if particle == gluon:
        result = [(0, 0)]
        for point, res in gluon:
            if res == False and result[-1][0] != point[0]:
                result.append((point))
        result = result[1:]

        for i in result:
            tempdf = lightdf.loc[lightdf["gg_factor"] == i[0]]
            if i[1] in list(tempdf.loc[tempdf["SM_deviation"] == False]["ll_factor"]):
                return np.round(1 - i[0], 5)

    elif particle == light:
        result = [(0, 0)]
        for point, res in light:
            if res == False and result[-1][1] != point[1]:
                result.append((point))
        result = result[1:]

        for i in result:
            tempdf = gluondf.loc[gluondf["ll_factor"] == i[1]]
            if i[0] in list(tempdf.loc[tempdf["SM_deviation"] == False]["gg_factor"]):
                return np.round(1 - i[1], 5)

    raise Exception("")


###############################################################################################################
lightPoints, lightDev = zip(*light)
g_light, l_light = zip(*lightPoints)

lightdf = pd.DataFrame()
lightdf["gg_factor"], lightdf["ll_factor"], lightdf["SM_deviation"] = g_light, l_light, lightDev
lightdf.sort_values(by=['ll_factor', 'gg_factor'], inplace=True)
lightdf.reset_index(drop=True, inplace=True)
points, res = zip(lightdf["gg_factor"], lightdf["ll_factor"]), list(lightdf["SM_deviation"])
light = zip(points, res)

################################################################################################################
gluonPoints, gluonDev = zip(*gluon)
g_gluon, l_gluon = zip(*gluonPoints)

gluondf = pd.DataFrame()
gluondf["gg_factor"], gluondf["ll_factor"], gluondf["SM_deviation"] = g_gluon, l_gluon, gluonDev
gluondf.sort_values(by=['gg_factor', 'll_factor'], inplace=True)
gluondf.reset_index(drop=True, inplace=True)
points, res = zip(gluondf["gg_factor"], gluondf["ll_factor"]), list(gluondf["SM_deviation"])
gluon = zip(points, res)

#########################################################################################
light_new = []

light_hori_lower = 1 - part_calc(gluon)
light_hori_upper = 1 + part_calc(gluon)
gluon_vert_left = light_hori_lower
gluon_vert_right = light_hori_upper

gluon_hori_lower = 1 - part_calc(light)
gluon_hori_upper = 1 + part_calc(light)
light_vert_left = gluon_hori_lower
light_vert_right = gluon_hori_upper

for i in light:
    i = list(i)
    if i[0][0] < light_hori_lower or i[0][0] > light_hori_upper:
        i[1] = True
    if i[0][1] < light_vert_left or i[0][1] > light_vert_right:
        i[1] = True
    light_new.append(i)

light = light_new

gluon_new = []

light_hori_lower = 1 - part_calc(gluon)
light_hori_upper = 1 + part_calc(gluon)
gluon_vert_left = light_hori_lower
gluon_vert_right = light_hori_upper

gluon_hori_lower = 1 - part_calc(light)
gluon_hori_upper = 1 + part_calc(light)
light_vert_left = gluon_hori_lower
light_vert_right = gluon_hori_upper

for i in gluon:
    i = list(i)
    if i[0][0] < gluon_vert_left or i[0][0] > gluon_vert_right:
        i[1] = True
    if i[0][1] < gluon_hori_lower or i[0][1] > gluon_hori_upper:
        i[1] = True
    gluon_new.append(i)

gluon = gluon_new

gcolours = {True: "red", False: "blue"}
lcolours = {True: "red", False: "blue"}


fig, axs = plt.subplots(1, 2, figsize=(18,8))

plt.suptitle("Thrust Minor", fontsize=24)

axs[0].scatter(gluondf["gg_factor"], gluondf["ll_factor"], s=10, c=gluondf["SM_deviation"].map(gcolours), alpha=0.5, label="")
axs[0].scatter(1, 1, s=50, color="green", label="Standard model", marker="o")
axs[0].axvline(1 - part_calc(gluon), color="g", label = "Constraint: +/- {}%".format(part_calc(gluon)*100))
axs[0].axvline(1 + part_calc(gluon), color="g")
axs[0].set_title("Gluon Range")
axs[0].set_ylabel("Light Quark Factor"); axs[0].set_xlabel("Gluon Factor")
axs[0].set_xlim(grange); axs[0].set_ylim(lrange)
axs[0].set_xticks(np.linspace(grange[0], grange[1], 11)); axs[0].set_yticks(np.linspace(lrange[0], lrange[1], 11))
axs[0].grid()
axs[0].legend(loc="upper right")

axs[1].scatter(lightdf["ll_factor"], lightdf["gg_factor"], s=10, c=lightdf["SM_deviation"].map(lcolours), alpha=0.5, label="")
axs[1].scatter(1, 1, s=50, color="green", label="Standard model", marker="o")
axs[1].axvline(1 - part_calc(light), color="g", label = "Constraint: +/- {}%".format(np.round(part_calc(light),3)*100))
axs[1].axvline(1 + part_calc(light), color="g")
axs[1].set_title("Light Range")
axs[1].set_xlabel("Light Quark Factor"); axs[1].set_ylabel("Gluon Factor")
axs[1].set_ylim(grange); axs[1].set_xlim(lrange)
axs[1].set_yticks(np.linspace(grange[0], grange[1], 11)); axs[1].set_xticks(np.linspace(lrange[0], lrange[1], 11))
axs[1].grid()
axs[1].legend(loc="upper right")

plt.show()

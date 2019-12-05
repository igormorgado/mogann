import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn
from matplotlib.patches import Patch
from matplotlib import animation
from deap import tools

def plot_cv_indices(cv, X, y, groups, ax, cmap_cv, cmap_data, lw=15):
    """Create a sample plot for indices of a cross-validation object."""

    n_splits = cv.n_splits

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=groups, marker='_', lw=lw, cmap=cmap_data)


    # Formatting
    yticklabels = list(range(n_splits)) + ['Class']
    ax.set(yticks=np.arange(n_splits+1) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+1.2, -.2], xlim=[0, len(y)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


def plot_deviation(dataset, ax, color=None, label=None, linewidth=1):
    x = np.arange(dataset.shape[1])
    mean = dataset.mean(axis=0)
    std = dataset.std(axis=0)
    plot = ax.plot(x, mean)
    fill = ax.fill_between(x, mean-std, mean+std, alpha=.2)
    if color is not None:
        plot[0].set_color(color)
        fill.set_color(color)

    if label is not None:
        plot[0].set_label(label)

    return [ax, plot[0], fill]


def animate(frame_index, logbook, ax):
    colors = seaborn.color_palette('Set1', n_colors=4)
    # colors = [ 'blue', 'red', 'green', 'purple' ]
    ax.clear()
    fronts = tools.emo.sortLogNondominated(logbook.select('pop')[frame_index],
            len(logbook.select('pop')[frame_index]))

    for i, (inds, color) in enumerate(zip(fronts, colors)):
        par = [tb.evaluate(ind) for ind in inds]
        df = pd.DataFrame(par)
        df.plot(ax=ax, kind='scatter', label=f'Front {i+1}', 
                x=df.columns[0], y=df.columns[1], alpha=.45, color=color, legend=False)

        patches, labels = ax.get_legend_handles_labels()
        ax.legend(patches, labels, loc='lower left')

    ax.set_xlim([0,1])
    ax.set_ylim([-60,0])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ticks = ax.get_yticks()
    ax.set_yticklabels([int(abs(tick)) for tick in ticks])

    ax.set_xlabel('Acc')
    ax.set_ylabel('#Attrs')
    ax.set_title(f't = {frame_index}: Pareto Front')
    #ax.set_tight_layout()

    return []

def plot_population_fitness(population, axes, color=None, alpha=None, sizes=None, label=None, clear=False): 
    """Plot a population fitness given optional parameters"""
    population_fitness = np.array([ p.fitness.values for p  in population])
    x = population_fitness[:, 0]
    y = population_fitness[:, 1]

    # FUCH
    axes.set_xlim([0,1])
    axes.set_ylim([-60,0])

    if clear:
        axes.clear()

    sct = axes.scatter(x, y)
    if color is not None:
        sct.set_color(color)
    if alpha is not None:
        sct.set_alpha(alpha)
    if sizes is not None:
        sct.set_sizes(sizes)
    if label is not None:
        sct.set_label(label)
    axes.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ticks = axes.get_yticks()
    axes.set_yticklabels([int(abs(tick)) for tick in ticks])
    axes.set_xlabel('Acurácia')
    axes.set_ylabel('#Atributos')

    return  sct


def plot_feature_histogram(data, axes, normalized=False):
    dset = np.copy(data)
    if normalized:
        dset = dset/np.sum(dset)
    axes.bar(np.arange(len(dset)), dset)
    axes.set_xlabel('Atributo')
    axes.set_title('Ocorrência de atributos')


def plot_feature_selection(data, axes):
    img = axes.imshow(data, cmap='Greys', aspect='auto', origin='lower')
    axes.set_xlabel('Atributo')
    axes.set_ylabel('Geração')
    cbar = plt.colorbar(img)
    cbar.ax.set_ylabel('%Atributo')
    axes.set_title('Atributos escolhidos por geração')
        
def plot_pareto_fronts(pop, axes, nfronts=5, showall=True):
    colors = seaborn.color_palette('Set1', n_colors=nfronts)
    fronts = tools.sortLogNondominated(pop, k=len(pop))

    fitnesses = np.array([ ind.fitness.values for ind in pop ])
    axes.scatter(fitnesses[:,0], fitnesses[:,1], color='black', sizes=[2], alpha=.5)

    for i, (color, front) in reversed(list(enumerate(zip(colors, fronts)))):
        fitnesses = np.array([ ind.fitness.values for ind in front ])
        #axes.scatter(fitnesses[:,0], fitnesses[:,1], color=color, sizes=[6], label=f'Frente {i+1}')
        axes.plot(fitnesses[:,0], fitnesses[:,1], marker='o', ms=2, color=color, label=f'Frente {i+1}')


# 
# 
# plt.close('all')
# fig, ax = plt.subplots(1, figsize=(6,4), dpi=150)
# for i, (color, inds) in enumerate(zip(colors, fronts)):
#     par = [tb.evaluate(ind) for ind in inds]
#     df = pd.DataFrame(par)  #PRECISO?
#     df.plot(ax=ax, kind='scatter', label=f'Front {i+1}', x=df.columns[0], y=df.columns[1], color=color)
# 
# ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# ticks = ax.get_yticks()
# ax.set_yticklabels([int(abs(tick)) for tick in ticks])
# plt.xlabel('Acc')
# plt.ylabel('#Attrs')
# plt.title('Pareto Front')
# plt.tight_layout()
# #plt.xlim([0,1])
# #plt.ylim([-60,0])
# 
# 
# #plt.legend()



#plt.close('all')
#fig = plt.figure(figsize=(6,5), dpi=150)
#ax = fig.gca()
#from matplotlib import animation
#anim = animation.FuncAnimation(fig, lambda i: animate(i, logbook, ax),
#        frames=len(logbook), interval=120, blit=True)
#
#anim.save('pareto.mp4')
#
#
#
#
# 
#fronts = tools.sortLogNondominated(pop, k=len(pop))
#plt.close('all')
#for ind in fronts[0]:
#    plt.plot(ind.fitness.values[1], ind.fitness.values[0], 'k.', ms=3, alpha=.5)


# plt.close('all')
# for ind in pop:
#     plt.plot(ind.fitness.values[1], ind.fitness.values[0], 'k.', ms=3, alpha=.5)
# 
# for color, front in zip(colors, fronts):
#     for i, ind in enumerate(front):
#         plt.plot(ind.fitness.values[1], ind.fitness.values[0], 'o', color=color, ms=4)
# 
# for ind in fronts[0]:
#     plt.plot(ind.fitness.values[1], ind.fitness.values[0], 'o', color='black', ms=5)
# 
# plt.ylim([0,1])
# plt.xlim([-60,0])
# plt.xlabel('#Attrs')
# plt.ylabel('Acc')
# plt.title('Pareto Front')
# 
# 
# plt.close('all')
# fig, ax = plt.subplots(1, figsize=(6,4), dpi=150)
# for i, (color, inds) in enumerate(zip(colors, fronts)):
#     par = [tb.evaluate(ind) for ind in inds]
#     df = pd.DataFrame(par)  #PRECISO?
#     df.plot(ax=ax, kind='scatter', label=f'Front {i+1}', x=df.columns[0], y=df.columns[1], color=color)
# 
# ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# ticks = ax.get_yticks()
# ax.set_yticklabels([int(abs(tick)) for tick in ticks])
# plt.xlabel('Acc')
# plt.ylabel('#Attrs')
# plt.title('Pareto Front')
# plt.tight_layout()
# #plt.xlim([0,1])
# #plt.ylim([-60,0])
# 
# 
# #plt.legend()


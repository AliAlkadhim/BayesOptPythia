from configs import *
from BayesOpt_utils import *


import scienceplots
import matplotlib.pyplot as plt
import matplotlib as mp
plt.style.use(['science', 'notebook', 'grid'])

# Then set custom font configurations
FONTSIZE = 20
font = {
    'family': 'serif',
    'weight': 'normal',
    'size': FONTSIZE
}
mp.rc('font', **font)

def compare_uniform_sobol(PARAM_DICT,size):
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    uniform_candidates = make_x_candidates(PARAM_DICT,size).detach().numpy()
    a_uniform = uniform_candidates[:,0]
    b_uniform = uniform_candidates[:,1]
    ax[0].scatter(a_uniform, b_uniform)
    ax[0].set_title('Uniform MC')

    sobol_candidates = make_sobol_candidates(PARAM_DICT,size)
    sobol_candidates=sobol_candidates.detach().numpy()
    a_sobol=sobol_candidates[:,0]
    b_sobol=sobol_candidates[:,1]
    ax[1].scatter(a_sobol, b_sobol)
    ax[1].set_title('Sobol')

    for i in range(2):
        ax[i].set_xlabel('aLund')
        ax[i].set_ylabel('bLund')
    plt.tight_layout()
    plt.show()




def plot_model_param(model,param, ax, filter_observed_data = False,  set_xy_lim=False):
    train_x = model.train_inputs[0].numpy()
    param_prefix = get_param_prefix(param)
    full_param_name = param_prefix + ':' + param
    param_index = param_names.index(full_param_name)

    train_x_param = train_x[:,param_index]
    train_y = model.train_targets.numpy()

    if train_x.shape[1] == 1 or train_x.ndim ==1:
        x_star = torch.linspace(train_x.min(), train_x.max(), 2000)
    else:
        # x_star = make_x_candidates(PARAM_DICT,200)
        # x_star  = make_multidim_xstar(model, param,2000)

        mean_values = train_x.mean(axis=0)
        x_star = torch.linspace(train_x.min(axis=0)[param_index],
                                train_x.max(axis=0)[param_index],
                                2000).unsqueeze(1)
        # Create a grid where only the parameter of interest varies
        other_params = mean_values.clone()
        other_params = torch.tensor(other_params).repeat(2000, 1)
        other_params[:, param_index] = x_star.squeeze()
        x_star = other_params

    model.eval()
    model.likelihood.eval()
    with torch.no_grad():
        predictive_distribution = model.predict(x_star)
        lower, upper = predictive_distribution.confidence_region()
        pred = predictive_distribution.mean.numpy()

    if train_x.shape[1] == 1 or train_x.ndim ==1:
        x_star_param =x_star.numpy()
    else:
      x_star_param =x_star[:,param_index].numpy()
    ax.plot(x_star_param, pred, label='GP Mean Prediction', color='red')
    ax.fill_between(x_star_param, lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)

    if filter_observed_data:
        train_x_param_other_prams_close_to_1, train_y_all_other_params_close_to_1 = filter_data_other_params_close_to_one(param, train_x, train_y)
        ax.plot(train_x_param_other_prams_close_to_1, train_y_all_other_params_close_to_1, 'k*', label='Observed Data', alpha=0.5)
    else:
        ax.plot(train_x_param, train_y, 'k*', label='Observed Data', alpha=0.5)
    # y_upper = upper.detach().numpy().max() * 1.5
    # y_lower = 0#lower.detach().numpy().max()
    # ax.set_ylim(y_lower, y_upper)
    ax.legend()
    ax.set_xlabel(param)
    ax.set_ylabel(r'$\lambda$')
    if set_xy_lim:
      ax.set_xlim(PARAM_DICT[full_param_name])
      if len(PARAM_DICT) <= 2:
        ax.set_ylim(-4,10)
      else:
        ax.set_ylim(-4,60)
    plt.tight_layout()



def plot_all(model,dirname, set_xy_lim=False, save_fig=False):
  if num_params <= 3:
    n_rows = 1
    fig, axs = plt.subplots(n_rows,1,figsize=(17,14))
  else:
    n_rows = num_params//3
    fig, axs = plt.subplots(n_rows,3,figsize=(17,14))
    axs=axs.ravel()
    
  pram_postfixs = [p.split(':')[1] for p in param_names]
  for axind, ax in enumerate(axs):
    plot_model_param(model, pram_postfixs[axind], ax,filter_observed_data = False,  set_xy_lim=set_xy_lim)
  plt.tight_layout()
  if not save_fig:
    plt.show()
  else:
    plt.savefig(os.path.join(dirname, 'parmetera_BO_plot.pdf'))




# if __name__ == '__main__':
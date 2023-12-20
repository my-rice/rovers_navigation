import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import numpy as np
import json
import scipy.stats as st
import scipy.stats as st
import matplotlib.pyplot as plt 
import numpy as np

################# UTILITIES #####################
control_space_size = 3

U_space_1 = np.array(np.linspace((-0.5),(0.5),control_space_size))
U_space_2 = np.array(np.linspace((-0.5),(0.5),control_space_size))
time_step = 0.033

X_LIMITS = [-1.5, 1.5]
Y_LIMITS = [-1, 1]
EPSILON_LIMITS = 0.0
X_SAMPLES = 100
Y_SAMPLES = 100
def model_step(x,velocities,time_step):
    poses = np.zeros((2,1))
    poses[0] = x[0] + time_step*velocities[0]
    poses[1] = x[1] + time_step*velocities[1]
    return(poses)

def my_logpdf(x, u, covar):
    k = len(x)  # dimension
    a = np.transpose(x - u)
    b = np.linalg.inv(covar)
    c = x - u
    d = np.matmul(a, b)
    e = np.matmul(d, c)
    numer = np.exp(-0.5 * e)
    f = (2 * np.pi)**k
    g = np.linalg.det(covar)
    denom = np.sqrt(f * g)
    pdf = numer / denom
    return pdf
def evaluate_2D_func(func,args={}): 
    # This function evaluates a function on the points of the robotarium grid
    X_axis = np.linspace(X_LIMITS[0]-EPSILON_LIMITS, X_LIMITS[1]+EPSILON_LIMITS, X_SAMPLES)
    Y_axis = np.linspace(Y_LIMITS[0]-EPSILON_LIMITS, Y_LIMITS[1]+EPSILON_LIMITS, Y_SAMPLES)
    Z = np.zeros((X_SAMPLES,Y_SAMPLES))
    for i in range(Y_SAMPLES):
        for j in range(X_SAMPLES):
            state = np.array([X_axis[j],Y_axis[i]])
            Z[j,i] = func(state,**args)
    return Z,X_axis,Y_axis
def plot_heatmap(cost, title='',cm='coolwarm',vmin=None,vmax=None, up_arrow=False):
    X_axis = np.linspace(X_LIMITS[0]-EPSILON_LIMITS, X_LIMITS[1]+EPSILON_LIMITS, X_SAMPLES)
    Y_axis = np.linspace(Y_LIMITS[0]-EPSILON_LIMITS, Y_LIMITS[1]+EPSILON_LIMITS, Y_SAMPLES)
    X_grid, Y_grid = np.meshgrid(X_axis, Y_axis)

    fig = plt.figure(figsize=(13,10))
    ax = fig.add_subplot()
    ax.set_xlabel('X [m]', labelpad=20)
    ax.set_ylabel('Y [m]', labelpad=20)
    ax.set_title(title, fontdict={'fontsize': 20})

    # Plot the heatmap of the cost function
    pcolormesh = ax.pcolormesh(X_grid, Y_grid, cost.T, cmap=cm, shading='auto', zorder=1, vmin=vmin, vmax=vmax)

    # Add a colorbar
    if up_arrow:
        fig.colorbar(pcolormesh, ax=ax, extend='max')
    else:
        fig.colorbar(pcolormesh, ax=ax)

    # Create a rectangle
    rectangle = Rectangle((X_LIMITS[0], Y_LIMITS[0]), X_LIMITS[1]-X_LIMITS[0], Y_LIMITS[1]-Y_LIMITS[0], fill=False, color="black", linewidth=1, zorder=3)
    # Add the rectangle to the plot
    ax.add_patch(rectangle)
    return ax
################# COST FUNCTIONS #####################
def state_cost(state,goal_points,obs_points):
    v = np.array([0.02, 0.02], dtype=np.float32)
    covar = np.diag(v)
    gauss_sum = 0
    for i in range(np.size(obs_points,axis=1)):
        gauss_sum += 20*my_logpdf(state[:2],obs_points[:2,i],covar)
    cost = 30*((state[0]-goal_points[0])**2 + (state[1]-goal_points[1])**2) + gauss_sum + 10*(np.exp(-0.5*((state[0]-(-1.5))/0.02)**2)/(0.02*np.sqrt(2*np.pi))
                + np.exp(-0.5*((state[0]-1.5)/0.02)**2)/(0.02*np.sqrt(2*np.pi)) + np.exp(-0.5*((state[1]-1.0)/0.02)**2)/(0.02*np.sqrt(2*np.pi))
                + np.exp(-0.5*((state[1]-(-1.0))/0.02)**2)/(0.02*np.sqrt(2*np.pi)))
    return(cost)

def state_cost_estimated_proposed(state,goal_points,obs_points,weights):
    # This function evaluates the estimated cost function on a given state
    # Feature 0 is the goal feature
    # Features 1 to len(obs_points[0]) are the obstacle features
    # Features len(obs_points[0])+1 to len(obs_points[0])+5 are the vertical walls features
    # Features len(obs_points[0])+6 to len(obs_points[0])+10 are the horizontal walls features
    barrier_variance=0.05
    v = np.array([0.025, 0.025], dtype=np.float32)
    covar = np.diag(v)
    gauss_sum = 0
    for i in range(np.size(obs_points,axis=1)):
        gauss_sum += -weights[:,i+2]*my_logpdf(state[:2],obs_points[:2,i],covar)
        
    a= -weights[:,0]* ((state[0]-goal_points[0])**2 + (state[1]-goal_points[1])**2 )
    cost = -weights[:,1]* (1/np.sqrt(((state[0]-goal_points[0])**2 + (state[1]-goal_points[1])**2 +0.1)) ) + gauss_sum +a
    x_walls = np.linspace(X_LIMITS[0], X_LIMITS[1], 5) 
    y_walls = np.linspace(Y_LIMITS[0], Y_LIMITS[1], 5) 
     
    x_wall = lambda x: np.exp(-0.5*((state[0]-x)/barrier_variance)**2)/(barrier_variance*np.sqrt(2*np.pi))  
    y_wall = lambda y: np.exp(-0.5*((state[1]-y)/barrier_variance)**2)/(barrier_variance*np.sqrt(2*np.pi))  
     
    for i,x in enumerate(x_walls): 
        cost += -weights[:,i+2+np.size(obs_points,axis=1)]*x_wall(x) 
         
    for i,y in enumerate(y_walls): 
        cost += -weights[:,i+2+np.size(obs_points,axis=1)+5]*y_wall(y) 
    return(cost)

def state_cost_estimated(state,goal_points,obs_points,weights):
    v = np.array([0.025, 0.025], dtype=np.float32)
    covar = np.diag(v)

    gauss_sum = 0

    for i in range(np.size(obs_points,axis=1)):
        gauss_sum += -weights[:,i+1]*my_logpdf(state[:2],obs_points[:2,i],covar)

    cost = -weights[:,0]*((((state[0]-goal_points[0])**2 + (state[1]-goal_points[1])**2))) + gauss_sum
    return(cost)

################# CONTROL FUNCTIONS #####################
def Control_step(state,U_space_1,U_space_2,goal_points,obs_points, cost_func, use_interpolated_function=False):
    target_pf = 1/control_space_size**2 # Uniform pf q(u_k|x_k-1)
    time_step = 0.033 # The Robotarium time-step
    pf = np.zeros((control_space_size,control_space_size)) #Initialize pf
    for i in range(control_space_size):
        for j in range(control_space_size):
            next_state = model_step(state,[U_space_1[i],U_space_2[j]],time_step)
            cov = np.array([[0.001, 0.0002], [0.0002, 0.001]])
            f = st.multivariate_normal(next_state.reshape((2,)),cov)
            N_samples = 20
            next_sample = f.rvs(N_samples) 
            cost=0
            for k in range(N_samples):
                if use_interpolated_function:
                    cost += cost_func(next_sample[k,:])/N_samples #interpolated cost does not need goal and obs points
                else:
                    cost += cost_func(next_sample[k,:],goal_points,obs_points)/N_samples
            log_DKL = np.exp(f.entropy()-cost).item() #item() function is called to cast the ndarray to a scalar which avoids a warning
            pf[i,j] = log_DKL
    S2 = np.sum(pf)
    pf = np.divide(pf,S2)
    flat = pf.flatten()
    sample_index = np.random.choice(a=flat.size, p=flat)
    adjusted_index = np.unravel_index(sample_index, pf.shape) 
    action = np.reshape(np.array([U_space_1[adjusted_index[0]],U_space_2[adjusted_index[1]]]),(2,1))
    return(action)

COST_FUNCTIONS = {
    'state_cost': {
        'function': state_cost
    },
    'state_cost_estimated_proposed': {
        'function': state_cost_estimated_proposed,
        'obs_feature_points': np.array(np.mat('0 0 0 0 0 0.8 0.8 0.8 0.8 0.8 -0.8 -0.8 -0.8 -0.8 -0.8 1.2 1.2 1.2 1.2 1.2 0.4 0.4 0.4 0.4 0.4 -0.4 -0.4 -0.4 -0.4 -0.4 -1.2 -1.2 -1.2 -1.2 -1.2; -0.8 -0.4 0 0.4 0.8 -0.8 -0.4 0 0.4 0.8 -0.8 -0.4 0 0.4 0.8 -0.8 -0.4 0 0.4 0.8 -0.8 -0.4 0 0.4 0.8 -0.8 -0.4 0 0.4 0.8 -0.8 -0.4 0 0.4 0.8; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ')),
    },
    'state_cost_estimated': {
        'function': state_cost_estimated,
        'obs_feature_points':np.array(np.mat('0 0 0 0 0 0.8 0.8 0.8 0.8 0.8 -0.8 -0.8 -0.8 -0.8 -0.8;-0.8 -0.4 0 0.4 0.8 -0.8 -0.4 0 0.4 0.8 -0.8 -0.4 0 0.4 0.8;0 0 0 0 0 0 0 0 0 0 0 0 0 0 0')),
    }
}

SCENARIOS = {
    'A': {
        'goal_points': np.array(np.mat('+1.3; -0.3; 0')), 
        'obs_points': np.array(np.mat('0.4 -0.8 -0.4 0.7 0.7; -0.5 0.4 -0.8 0.8 -0.8; 0 0 0 0 0')),
        'initial_conditions': np.array(np.mat('-0.9;-0.75; 0')),
    },
    'baseline': {
        'goal_points': np.array(np.mat('-1.4; -0.8; 0')), 
        'obs_points': np.array(np.mat('0 0 0 0 0;0.2 0.4 0.6 0.8 -0.8;0 0 0 0 0')),
        'initial_conditions': np.array(np.mat('1.4;0.9; 0')),
    },
}

config = {
    "cost_function": "state_cost_estimated",
    "scenario": "baseline",
    "weights": "weights_baseline_cost_baseline_scenario.npy"
}


if __name__ == "__main__":
    # Read json file
    # with open('config.json') as json_file:
    #     config = json.load(json_file)
    
    cost_function_family = COST_FUNCTIONS[config['cost_function']]['function']
    scenario = SCENARIOS[config['scenario']]
    goal_points = scenario['goal_points']
    obs_points = scenario['obs_points']
    weights = np.load(config['weights'])
    obs_feature_points = COST_FUNCTIONS[config['cost_function']]['obs_feature_points']
    
    cost_function = lambda state,goal_points,obs_points: cost_function_family(state,goal_points,obs_feature_points,weights)
    
    # print('Cost function: ', cost_function)
    # print('Scenario: ', scenario)
    # print('Weights: ', weights)
    
    N = 1 #Amount of robots per simulation

    r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=np.copy(scenario['initial_conditions']), sim_in_real_time=False)
    si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion() #Converts single integrator inputs to unicycle inputs (low-level controller)
    _, uni_to_si_states = create_si_to_uni_mapping()

    x = r.get_poses()
    x_si = uni_to_si_states(x)

                # Plotting Parameters
    CM = np.random.rand(N+10,3) # Random Colors
    goal_marker_size_m = 0.15
    obs_marker_size_m = 0.15
    marker_size_goal = determine_marker_size(r,goal_marker_size_m)
    marker_size_obs = determine_marker_size(r,obs_marker_size_m)
    font_size = determine_font_size(r,0.1)
    line_width = 5

    # Create Goal Point Markers
    #Text with goal identification
    goal_caption = ['G{0}'.format(ii) for ii in range(goal_points.shape[1])]
    #Plot text for caption
    goal_points_text = [r.axes.text(goal_points[0,ii], goal_points[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-2)
    for ii in range(goal_points.shape[1])]
    goal_markers = [r.axes.scatter(goal_points[0,ii], goal_points[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-2)
    for ii in range(goal_points.shape[1])]

    #Text with goal identification
    obs_caption = ['OBS{0}'.format(ii) for ii in range(obs_points.shape[1])]
    #Plot text for caption
    obs_points_text = [r.axes.text(obs_points[0,ii], obs_points[1,ii], obs_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-2)
    for ii in range(obs_points.shape[1])]
    obs_markers = [r.axes.scatter(obs_points[0,ii], obs_points[1,ii], s=marker_size_obs, marker='s', facecolors='none',edgecolors=CM[ii+1,:],linewidth=line_width,zorder=-2)
    for ii in range(obs_points.shape[1])]
    
    r.step()

    i = 0
    while (np.size(at_pose(np.vstack((x_si,x[2,:])), goal_points, position_error=0.15,rotation_error=100)) != N):
        x = r.get_poses()
        x_si = uni_to_si_states(x)
        
        cov = np.array([[0.001, 0.0002], [0.0002, 0.001]])
        x_pdf = st.multivariate_normal(x_si.reshape((2,)),cov)
        x_sample = x_pdf.rvs() #Noisy state
        
        for j in range(goal_points.shape[1]):
            goal_markers[j].set_sizes([determine_marker_size(r, goal_marker_size_m)])

        for j in range(obs_points.shape[1]):
            obs_markers[j].set_sizes([determine_marker_size(r, obs_marker_size_m)])
        
        dxi = Control_step(x_sample, U_space_1, U_space_2, goal_points, obs_points, cost_function, False)
        dxu = si_to_uni_dyn(dxi, x)
        r.set_velocities(np.arange(N), dxu)
        # Iterate the simulation
        r.step()

    #Call at end of script to print debug information and for your script to run on the Robotarium server properly
    r.call_at_scripts_end()
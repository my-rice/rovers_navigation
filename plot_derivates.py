import matplotlib.pyplot as plt
import numpy as np


bound1 = -1.5
bound2 = 1.5
bound3 = -1.0
bound4 = 1.0

resolution = 0.1

goal_points = np.array(np.mat('0; 0; 0'))
obs_points = np.array(np.mat('-0.3 -0.3 -0.3 0 0.3 0.3 0.3; -0.3 0 0.3 0.3 0.3 0 -0.3; 0 0 0 0 0 0 0'))
def V(x,y):
    return x**2+y**2

def my_cilinder(x,u):
    r = 0.4
    k = 1000
    height = 100
    
    temp = height*(-1/(1+np.exp(-k*((x[0]-u[0])**2+(x[1]-u[1])**2-(r**2)))) + 1)
    return temp

def my_logpdf(x, u, covar):
    k = len(x)  # dimension
    #print("u.shape\n",u.shape,"x.shape\n",x.shape)
    #print("u\n",u)
    #print("x\n",x)
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


def state_cost(x,y):
    state = np.array([x,y])

    v = np.array([0.02, 0.02], dtype=np.float32)
    covar = np.diag(v)
    gauss_sum = 0

    for i in range(np.size(obs_points,axis=1)):
        gauss_sum += 20*my_logpdf(state[:2],obs_points[:2,i],covar)

    cost = 30*((state[0]-goal_points[0])**2 + (state[1]-goal_points[1])**2) + gauss_sum + 10*(np.exp(-0.5*((state[0]-(-1.5))/0.02)**2)/(0.02*np.sqrt(2*np.pi))
                + np.exp(-0.5*((state[0]-1.5)/0.02)**2)/(0.02*np.sqrt(2*np.pi)) + np.exp(-0.5*((state[1]-1.0)/0.02)**2)/(0.02*np.sqrt(2*np.pi))
                + np.exp(-0.5*((state[1]-(-1.0))/0.02)**2)/(0.02*np.sqrt(2*np.pi)))
    return(cost)

def state_cost_with_additional_term(x,y):
    # print("\nx\n",x)
    # print("\ny\n",y)
    state = np.array([x,y])

    v = np.array([0.02, 0.02], dtype=np.float32)
    covar = np.diag(v)
    gauss_sum = 0

    for i in range(np.size(obs_points,axis=1)):
        gauss_sum += 20*my_logpdf(state[:2],obs_points[:2,i],covar)


    state_cost = ((state[0]-goal_points[0])**2 + (state[1]-goal_points[1])**2) -1/((state[0]-goal_points[0])**2 + (state[1]-goal_points[1])**2 +0.1)
    cost = 30*state_cost + gauss_sum + 10*(np.exp(-0.5*((state[0]-(-1.5))/0.02)**2)/(0.02*np.sqrt(2*np.pi))
                + np.exp(-0.5*((state[0]-1.5)/0.02)**2)/(0.02*np.sqrt(2*np.pi)) + np.exp(-0.5*((state[1]-1.0)/0.02)**2)/(0.02*np.sqrt(2*np.pi))
                + np.exp(-0.5*((state[1]-(-1.0))/0.02)**2)/(0.02*np.sqrt(2*np.pi)))
    return(cost)

def state_cost_with_additional_term_with_cilinder(x,y):
    state = np.array([x,y])

    v = np.array([0.02, 0.02], dtype=np.float32)
    covar = np.diag(v)
    gauss_sum = 0

    for i in range(np.size(obs_points,axis=1)):
        #print("state[:2]\n",state[:2])
        #print("obs_points[:2,i]\n",obs_points[:2,i])
        gauss_sum += 20*my_cilinder(state[:2],obs_points[:2,i])

    state_cost = ((state[0]-goal_points[0])**2 + (state[1]-goal_points[1])**2) -1/((state[0]-goal_points[0])**2 + (state[1]-goal_points[1])**2 +0.1)
    cost = 30*state_cost + gauss_sum + 10*(np.exp(-0.5*((state[0]-(-1.5))/0.02)**2)/(0.02*np.sqrt(2*np.pi))
                + np.exp(-0.5*((state[0]-1.5)/0.02)**2)/(0.02*np.sqrt(2*np.pi)) + np.exp(-0.5*((state[1]-1.0)/0.02)**2)/(0.02*np.sqrt(2*np.pi))
                + np.exp(-0.5*((state[1]-(-1.0))/0.02)**2)/(0.02*np.sqrt(2*np.pi)))
    return(cost)


def state_cost_with_abs_terms(state,goal_points,obs_points):
    v = np.array([0.02, 0.02], dtype=np.float32)
    covar = np.diag(v)
    gauss_sum = 0

    for i in range(np.size(obs_points,axis=1)):
        gauss_sum += 20*my_logpdf(state[:2],obs_points[:2,i],covar)

    state_cost = 60*(np.abs(state[0]-goal_points[0]) + np.abs(state[1]-goal_points[1])) -1/((state[0]-goal_points[0])**2 + (state[1]-goal_points[1])**2 +0.1)

    cost = state_cost + gauss_sum + 10*(np.exp(-0.5*((state[0]-(-1.5))/0.02)**2)/(0.02*np.sqrt(2*np.pi))
                + np.exp(-0.5*((state[0]-1.5)/0.02)**2)/(0.02*np.sqrt(2*np.pi)) + np.exp(-0.5*((state[1]-1.0)/0.02)**2)/(0.02*np.sqrt(2*np.pi))
                + np.exp(-0.5*((state[1]-(-1.0))/0.02)**2)/(0.02*np.sqrt(2*np.pi)))
    return(cost)

def plot_cost(cost_name='V'):
    x_axis = np.linspace(bound1,bound2,100)
    y_axis = np.linspace(bound3,bound4,100)
    X,Y=np.meshgrid(x_axis,y_axis)
    # print(X.shape)
    # print("X\n",X)
    # print(Y.shape)
    if cost_name == 'state_cost_with_additional_term':
        Z = np.array([[state_cost_with_additional_term(x,y) for x in x_axis] for y in y_axis] )
        #costs = np.array([[state_cost((x,y),goal_points,obs_points) for x in X_axis] for y in Y_axis] )
    elif cost_name == 'state_cost':
        Z = np.array([[state_cost(x,y) for x in x_axis] for y in y_axis] )
    elif cost_name == 'state_cost_with_additional_term_with_cilinder':
        Z = np.array([[state_cost_with_additional_term_with_cilinder(x,y) for x in x_axis] for y in y_axis] )
    elif cost_name == 'state_cost_with_abs_terms':
        Z = np.array([[state_cost_with_abs_terms(np.array([x,y]),goal_points,obs_points) for x in x_axis] for y in y_axis] )
    else:
        Z = V(X,Y)
    #print(Z.shape)
    Z = np.squeeze(Z)
    #print("Z\n",Z)

    grad_x, grad_y = np.gradient(Z, x_axis, y_axis)

    fig = plt.figure(figsize=(13,10))
    ax = fig.add_subplot()

    gradient_module = np.sqrt(grad_x**2 + grad_y**2)

    pcolormesh = ax.pcolormesh(X, Y, gradient_module, cmap='coolwarm', zorder=1, vmax=300)

    # Plot the level curves of the cost function
    contour= ax.contour(X, Y, gradient_module, cmap='coolwarm', linestyles='dashed', zorder=2, alpha=1, vmax=300) #vmin and vmax are used to make the contour lines more visible

    # Add a colorbar
    fig.colorbar(pcolormesh,ax=ax) 

    plt.savefig(cost_name+".png")
    plt.close()


if __name__ == "__main__":
    selected_cost = 'state_cost_with_abs_terms'
    plot_cost(selected_cost)
    print('Done')



    
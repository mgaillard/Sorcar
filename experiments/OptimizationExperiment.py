from Functions import CasadiFunction, generate_functions
from Optimizer import Optimizer

# TODO: activate the bounds in the optimizer in the Blender add-on
# TODO: create Blender example with two variables (two cube stack)
# TODO: benchmark the time needed for computing the gradient µvs the Hessian of a function
# TODO: run K-Medoid on the set of all optimal points
# TODO: find an order for the medoids
# TODO: try nonlinear-PCA on the set of points
# TODO: show a slider (1D or 2D) for exploring the solution set, and by changing it, show the position on the 2D plot
# TODO: try trust region methods for local optimization
# TODO: parallelize to speed up optimization, run the algorithm many time in parallel and aggregate results
# TODO: when running global optimization, start from random points with basinhopping
# TODO: try to propose more than one option when giving a delta solution or a proportional solution
# TODO: change the take step function for exploration so that it takes steps based on the Hessian
# TODO: add legend in the 2D plot for specific points
# TODO: animation of the optimization plot
# TODO: minimize the list of functions with different optimizers and show different animations

def main():
    functions = generate_functions()

    functions_to_optimize = [
        'rosen',
        'underdetermined_linear',
        'underdetermined_circle',
        'underdetermined_disk',
        'underdetermined_arm',
        'sorcar_cube_size_underdetermined'
    ]

    # Optimization of the functions
    for f in functions_to_optimize:
        optimizer = Optimizer(functions[f]['function'],
                              functions[f]['bounds'],
                              functions[f]['starting_point'])
        optimizer.optimize(400)
        print('Total optimization time: {} s'.format(optimizer.get_total_time()))
        optimizer.plot_2D()


if __name__ == "__main__":
    main()

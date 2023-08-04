from src.optimisers import LossFunction, optimise_with_GA, optimise_with_PSO
from src.model_management import SymmetricManager, UniformManager
from src.results_manager import ResultsManager
from src.face_model import GPModel, RBFModel
from src.graph_manager import GraphManager
import numpy as np



def show_sensor_layout(layout, model_manager, graph_manager):
    # Plots 3D graph + comparison pane
    positions = model_manager.get_all_positions()
    true_temperatures = model_manager.get_all_temp_values()
    model_temperatures = model_manager.get_all_model_temperatures(layout)
    
    if model_manager.is_symmetric():
        layout = model_manager.reflect_position(layout)
    else:
        layout = layout.reshape(-1, 2)
    for i, pos in enumerate(layout):
        layout[i] = model_manager.find_nearest_pos(pos)

    graph_manager.draw_double_3D_temp_field(
        positions, 
        true_temperatures, 
        model_temperatures
    )
    graph_manager.draw_comparison_pane(
        positions, 
        layout.reshape(-1, 2), 
        true_temperatures, 
        model_temperatures
    )



def show_pareto(graph_manager, is_symmetric=True):
    if is_symmetric == True:
        results_manager = ResultsManager('best_symmetric_setups.txt')
    else:
        results_manager = ResultsManager('best_uniform_setups.txt')
    
    numbers = results_manager.get_nums()
    results = []
    for num in numbers:
        results.append(results_manager.read_file(num)[0])
    graph_manager.draw_pareto(numbers, results)



def check_results(res, is_symmetric, num_sensors):
    if is_symmetric == True:
        results_manager = ResultsManager('best_symmetric_setups.txt')
    else:
        results_manager = ResultsManager('best_uniform_setups.txt')

    if res.F[0] < results_manager.read_file(num_sensors)[0]:
        print('\nSaving new record...')
        results_manager.write_file(num_sensors, res.F[0], list(res.X))
        results_manager.save_updates()



def optimise_sensor_layout(model_manager, graph_manager, num_sensors=10, time_limit='00:00:30'):
    # Optimises the sensor placement
    problem = LossFunction(num_sensors, model_manager)
    res = optimise_with_PSO(problem, time_limit)

    check_results(
        res, 
        model_manager.is_symmetric(), 
        num_sensors
    )
    graph_manager.draw_optimisation(res.history)
    show_sensor_layout(
        res.X, 
        model_manager, 
        graph_manager
    )
    print('\nResult:')
    print(res.X)



def show_best(graph_manager, model_manager, num):
    if model_manager.is_symmetric() == True:
        results_manager = ResultsManager('best_symmetric_setups.txt')
    else:
        results_manager = ResultsManager('best_uniform_setups.txt')
    
    loss, layout = results_manager.read_file(num)
    show_sensor_layout(
        np.array(layout), 
        model_manager, 
        graph_manager
    )











if __name__ == '__main__':
    graph_manager = GraphManager()
    symmetric_manager = SymmetricManager('temperature_field.csv', RBFModel)
    uniform_manager = UniformManager('temperature_field.csv', RBFModel)

    layout = np.array([0.012569, 0.0058103, 0.0088448, 0.0202931, 0.0041897, 0.0118448, 0.0079138, 0.0046034, 0.0088448, -0.0074655])
    #show_sensor_layout(layout, symmetric_manager, graph_manager)
    #optimise_sensor_layout(uniform_manager, graph_manager, 5, '00:10:00')
    #show_pareto(graph_manager, False)
    show_best(graph_manager, uniform_manager, 5)
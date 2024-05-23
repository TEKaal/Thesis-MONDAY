import argparse
import csv
from datetime import datetime
import optuna
import optuna.visualization as vis
from pymgrid.microgrid.reward_shaping import *
from pymgrid.microgrid.trajectory.stochastic import FixedLengthStochasticTrajectory
from RL_helpfunctions import *
from RL_helpfunctionDQN import *
from RL_visualizegrid import *
from RL_microgrid_environment import *
from RL_read_energy_data import *
from RL_connection_matrix import *
from RL_custom_Env import *
from Cluster_algorithm import *


def objective(trial):
    # Define the hyperparameters to tune
    dqn_episodes = trial.suggest_int('dqn_episodes', 400, 1000)
    dqn_batch_size = trial.suggest_int('dqn_batch_size', 32, 128)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    memory_size = trial.suggest_int('memory_size', 672*4, 672*8)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.9, 0.999)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    layer_size = trial.suggest_int('layer_size', 64, 256)

    output_folder = r"C:\Users\TesselAdmin\Downloads\Thesis-main\Thesis-main\output_optuna"
    microgrid_env = trial.study.user_attrs['microgrid_env']
    nr_steps = trial.study.user_attrs['nr_steps']
    dqn_evaluation_steps = trial.study.user_attrs['dqn_evaluation_steps']
    
    # Train the agent
    agent = train_dqn_agent(microgrid_env, output_folder, dqn_episodes, nr_steps, dqn_batch_size, learning_rate,
                            memory_size, num_layers, layer_size, epsilon_decay, gamma)

    microgrid_env.initial_step = 3360
    microgrid_env.final_step = 4992 # 52 mondays

    # Evaluate the agent
    average_reward, _ = evaluate_dqn_agent(microgrid_env, output_folder, agent, dqn_evaluation_steps, nr_steps)

    return average_reward

def save_arguments_to_csv(args, outputfolder):
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{outputfolder}\\arguments_{current_datetime}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Argument', 'Value'])
        for arg, value in vars(args).items():
            writer.writerow([arg, value])
    print(f"Arguments saved to {csv_filename}")

def main(args):
    save_arguments_to_csv(args, args.outputfolder)

    # Define the time variables
    nr_steps = args.nr_steps
    time_interval = args.time_interval

    # Load the case study and scenario files
    df_buildings, coordinates_buildings, horizontal_roof, ids_buildings, type_buildings = load_buildings_from_file(args.case_study_file)
    df_ders, coordinates_ders, ids_ders, type_der = load_DERs_from_file(args.scenario_file, ids_buildings)
    combined_df = concatenate_and_combine_columns(df_buildings, df_ders)

    # Print column names to check for unexpected characters or spaces
    print(combined_df.columns)

    # Load all the energy data
    Energy_consumption = process_energy_consumption_files(args.folder_path_loads, list(ids_buildings), time_interval)

    microgrid = create_microgrid(Energy_consumption, combined_df, df_buildings)
    print(type(microgrid))

    microgrid.trajectory_func = FixedLengthStochasticTrajectory(nr_steps)

    microgrid_env = CustomMicrogridEnv.from_microgrid(microgrid)

    microgrid_env.initial_step = 0
    microgrid_env.final_step = 3360 # 35 mondays

    microgrid_env.trajectory_func = FixedLengthStochasticTrajectory(nr_steps)

    # trained_agent_DQN = train_dqn_agent(microgrid_env, args.outputfolder, args.dqn_episodes, nr_steps,
    #                                                        args.dqn_batch_size, args.learning_rate, args.memory_size,  args.num_layers,
    #                                                        args.layers_size, args.epsilon_d, args.gamma)
    # print('TRAINING DQL DONE')

    # microgrid_env.initial_step = 3360
    # microgrid_env.final_step = 4992 # 52 mondays

    # evaluate_dqn_agent(microgrid_env,
    #                                       args.outputfolder,
    #                                       trained_agent_DQN,
    #                                       args.dqn_evaluation_steps,
    #                                       nr_steps)

    return microgrid_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run microgrid simulation.')

    # File name containing the loads
    folder_path_loads = r"Thesis\Final loads"
    case_study_file = r"Thesis\Buildings and scenarios\CS1.csv"
    scenario_file = r"Thesis\Buildings and scenarios\Scenario1.csv"
    output_file = r"C:\Users\tessel.kaal\OneDrive - Accenture\Thesis\Output training model"

    parser.add_argument('--outputfolder', type=str, default=output_file, help='Folder to save output files.')
    parser.add_argument('--folder_path_loads', type=str, default=folder_path_loads, help='Path to the folder containing load files.')
    parser.add_argument('--case_study_file', type=str, default=case_study_file, help='Path to the case study file.')
    parser.add_argument('--scenario_file', type=str, default=scenario_file, help='Path to the scenario file.')
    parser.add_argument('--nr_steps', type=int, default=96, help='Number of steps for the simulation.')
    parser.add_argument('--time_interval', type=int, default=15, help='Time interval in minutes.')
    parser.add_argument('--dqn_episodes', type=int, default=800, help='Number of episodes for DQN training.')
    parser.add_argument('--dqn_batch_size', type=int, default=64, help='Batch size for DQN training.')
    parser.add_argument('--dqn_evaluation_steps', type=int, default=50, help='Number of evaluation steps for DQN.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for DQN.')
    parser.add_argument('--memory_size', type=int, default=672*4, help='Memory allocation.')
    parser.add_argument('--num_layers', type=int, default=4, help='Neural network')
    parser.add_argument('--layers_size', type=int, default=64, help='Neural layer size')
    parser.add_argument('--epsilon_d', type=float, default=0.995, help='Memory allocation.')
    parser.add_argument('--gamma', type=float, default=1.0, help='Memory allocation.')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for Optuna optimization.')

    args = parser.parse_args()

    # Initialize the environment in the main function
    microgrid_env = main(args)

    # Use Optuna to find the best hyperparameters
    study = optuna.create_study(direction='maximize')
    print("hello")

    # Pass the environment and other static parameters to the study
    study.set_user_attr('microgrid_env', microgrid_env)
    study.set_user_attr('nr_steps', args.nr_steps)
    study.set_user_attr('dqn_evaluation_steps', args.dqn_evaluation_steps)

    study.optimize(objective, n_trials=args.n_trials)

    # Print the best hyperparameters found
    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)

    vis.plot_optimization_history(study)
    vis.plot_param_importances(study)
    vis.plot_slice(study)

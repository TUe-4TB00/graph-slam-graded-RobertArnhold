import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, gtsam.LevenbergMarquardtParams())

    # TODO: Perform the optimization and print the result
    result = optimizer.optimize()
    print("\nFinal Result:\n{}".format(result))

    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    best_pose = ""
    best_landmark = 0
    min_marginal = np.inf
    
    for pose_nr, pose_5 in pose_options.items():
        for landmark_option in [1, 2]:
            test_graph = gtsam.NonlinearFactorGraph(graph)
            test_estimate = gtsam.Values(initial_estimate)
            
            test_graph, test_estimate = add_pose(test_graph, test_estimate, pose_5)
            result = optimize(test_graph, test_estimate)

            test_graph = add_landmark_measurement(test_graph, result, pose_5, landmark_option)
            final_result = optimize(test_graph, test_estimate)
    
            # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
            marginals = gtsam.Marginals(test_graph, final_result)
            
            sum_covariances = (marginals.marginalCovariance(L(1)).sum() + marginals.marginalCovariance(L(2)).sum())
            
            if marginals.marginalCovariance(L(landmark_option)).sum() < min_marginal:
                min_marginal = sum_covariances
                best_pose = pose_nr
                best_landmark = landmark_option

                
    # Return the best combination found
    return best_pose, best_landmark, min_marginal





def minimize_errors(graph, initial_estimate, pose_options):
    # TODO: compute the sum of the errors and return it along with the best pose and landmark
    
    best_pose = ""       # chosen pose option
    best_landmark = 0    # chosen landmark (1 or 2)
    min_error = np.inf
    
    # TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
    list_of_errors = []

    for pose_nr, pose_5 in pose_options.items():
        
        #init lowest error for this pose option
        pose_lowest_error = np.inf
        pose_best_landmark = 0
        
        for landmark_option in [1, 2]:
            test_graph = gtsam.NonlinearFactorGraph(graph)
            test_estimate = gtsam.Values(initial_estimate)
            
            test_graph, test_estimate = add_pose(test_graph, test_estimate, pose_5)
            result = optimize(test_graph, test_estimate)

            test_graph = add_landmark_measurement(test_graph, result, pose_5, landmark_option)
            final_result = optimize(test_graph, test_estimate)
            
            # test_graph.error() to compute total sum of squared errors
            sum_of_errors = test_graph.error(final_result)
            
            if sum_of_errors < pose_lowest_error:
                pose_lowest_error = sum_of_errors
                pose_best_landmark = landmark_option
                
        list_of_errors.append(pose_lowest_error)
        
        if pose_lowest_error < min_error:
            min_error = pose_lowest_error
            best_pose = pose_nr
            best_landmark = pose_best_landmark

    # TODO: compute the sum of the errors and return it along with the best pose and landmark
    sum_of_errors = min_error

    return best_pose, best_landmark, sum_of_errors
import os
import sys
import json
import logging
import time
import yaml
import torch
import argparse
import gzip
import numpy                                   as np
import matplotlib.pyplot                       as plt
from   types                                   import SimpleNamespace
from   envs.matrix_env                         import construct_envs
from   agents.zeroshot.unigoal.agent           import UniGoal_Agent
from   agents.zeroshot.unigoal.map.bev_mapping import BEV_Map
from   agents.zeroshot.unigoal.graph.graph     import Graph
#from   agents.utils.disk_viz                   import DiskViz
from   utils.disk_viz                   import DiskViz

def get_config():
    # priority order: setting in this func > command line > yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",          default="agents/zeroshot/unigoal/configs/config_matrix.yaml", metavar="FILE", help="path to config file"    , type=str)
    parser.add_argument("--goal_type",            default="",                                           help="ins_image or text"      , type=str)
    parser.add_argument("--image_goal_path",      default="",                                           help="path for iamge input"   , type=str)
    parser.add_argument("--text_goal", default="",                                                      help=""                       , type=str)
    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    cli_args = vars(args)
    cli_args = {k: v for k, v in cli_args.items() if v not in ("", None)}
    config.update(cli_args)
    args     = config
    args     = SimpleNamespace(**args)

    args.map_size                         = args.map_size_cm // args.map_resolution
    args.global_width, args.global_height = args.map_size, args.map_size
    args.local_width                      = int(args.global_width / args.global_downscaling)
    args.local_height                     = int(args.global_height / args.global_downscaling)
    args.log_dir                          = os.path.join(args.dump_location, args.experiment_id, 'log')
    args.visualization_dir                = os.path.join(args.dump_location, args.experiment_id, 'visualization')
    args.device                           = torch.device("cuda:0" if args.cuda else "cpu")
    if args.cloud_api:
        args.api_key                          = os.getenv("DASHSCOPE_API_KEY")
    return args

def main():

    args = get_config()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.visualization_dir, exist_ok=True)
    logging.basicConfig( filename=os.path.join(args.log_dir, 'eval.log'), level=logging.INFO)
    logging.info(args)
    viz                 = DiskViz(output_root=args.visualization_dir)
    viz.preferred_first = ["rgb","exploration map","frontier map","dilated occupancy map","map | position | trajectory | goal"]

    BEV_map = BEV_Map(args)
    graph   = Graph(args)
    envs    = construct_envs(args)
    agent   = UniGoal_Agent(args, envs)

    BEV_map.init_map_and_pose()
    obs, rgbd, infos = agent.reset()

    BEV_map.mapping(obs,rgbd, infos,viz,0)

    local_goal = [args.local_width // 2, args.local_height // 2]
    goal_maps    = np.zeros((args.local_width, args.local_height))
    goal_maps[local_goal[0], local_goal[1]] = 1

    next_goal      = [(args.local_width//2) *3, (args.local_height // 2) *3] #in global map
    next_goal_maps = np.zeros(((args.local_width//2) *6, (args.local_height // 2) *6))
    next_goal_maps[next_goal[0], next_goal[1]] = 1

    agent_input               = {}
    agent_input['map_pred']   = BEV_map.local_map[0, 0, :, :].cpu().numpy()
    agent_input['exp_pred']   = BEV_map.local_map[0, 1, :, :].cpu().numpy()
    agent_input['pose_pred']  = BEV_map.planner_pose_inputs[0]
    agent_input['goal']       = goal_maps
    agent_input['exp_goal']   = goal_maps * 1
    agent_input['new_goal']   = 1
    agent_input['found_goal'] = 0
    agent_input['wait']       = False
    agent_input['sem_map']    = BEV_map.local_map[0, 4:11, :, :].cpu().numpy()

    graph.reset()

    graph.set_obj_goal(infos['goal_name'])

    obs, rgbd, done, infos = agent.step(agent_input)

    if args.goal_type == 'ins_image':
        graph.set_image_goal(infos['instance_imagegoal'])
        #print(infos['instance_imagegoal'])
    elif args.goal_type == 'text':
        infos['text_goal']     = {'intrinsic_attributes': args.text_goal, 'extrinsic_attributes': "in an apppartment"}
        #print(infos['text_goal'])
        graph.set_text_goal(infos['text_goal'])

    step = 0

    goal_stack = []
    goal_stack.append(next_goal)

    #global_goal = next_goal.copy()
    while True:

        BEV_map.mapping(obs,rgbd, infos,viz,step)
        navigate_steps = step
        graph.set_navigate_steps(navigate_steps)

        if not agent_input['wait'] and navigate_steps % 2 == 0:
            graph.set_observations(obs)
            graph.update_scenegraph(obs)


        BEV_map.update_intrinsic_rew(viz,step,next_goal_maps)
        if envs.info["goal_invalid"] or np.linalg.norm(np.array([BEV_map.local_row, BEV_map.local_col]) - np.array(local_goal)) < 10:
            if envs.info["goal_invalid"]:
                print("got invalid goal")
            envs.info["goal_invalid"]=False
            BEV_map.update_intrinsic_rew(viz,step,next_goal_maps)
            BEV_map.move_local_map()
            graph.set_full_map(BEV_map.full_map)
            graph.set_full_pose(BEV_map.full_pose)

            global_goal = graph.explore(BEV_map.local_map_boundary,viz,step)

            if global_goal is not None:
                goal_stack.append(global_goal)
                #print("len(goal_stack)=",len(goal_stack))
            else:
                goal_stack.pop()
                global_goal = goal_stack[-1]
                print("global goal popped")

            next_goal           = global_goal
            next_goal_maps[:,:] = 0
            next_goal_maps[next_goal[0], next_goal[1]] = 1

            global_goal = list(global_goal)
            local_goal[0] = global_goal[0] - BEV_map.local_map_boundary[0, 0]
            local_goal[1] = global_goal[1] - BEV_map.local_map_boundary[0, 2]

            if 0 <= local_goal[0] < args.local_width and 0 <= local_goal[1] < args.local_height:
                local_goal = local_goal
            else:
                import random

                if local_goal[0] < 0:
                    local_goal[0] = 0 + random.randint(1, 10)
                elif local_goal[0] >= args.local_width:
                    local_goal[0] = args.local_width - random.randint(1, 10)

                if local_goal[1] < 0:
                    local_goal[1] = 0 + random.randint(1, 10)
                elif local_goal[1] >= args.local_height:
                    local_goal[1] = args.local_height - random.randint(1, 10)



        goal_maps = np.zeros((args.local_width, args.local_height))
        goal_maps[local_goal[0], local_goal[1]] = 1
        exp_goal_maps = goal_maps.copy()

        agent_input                =  {}
        agent_input['map_pred']    =  BEV_map.local_map[0, 0, :, :].cpu().numpy()
        agent_input['exp_pred']    =  BEV_map.local_map[0, 1, :, :].cpu().numpy()
        agent_input['pose_pred']   =  BEV_map.planner_pose_inputs[0]
        agent_input['goal']        =  goal_maps
        agent_input['exp_goal']    =  exp_goal_maps
        agent_input['new_goal']    =  False
        #agent_input['found_goal'] =  found_goal
        agent_input['wait']        =  False
        agent_input['sem_map']     =  BEV_map.local_map[0, 4:11, :, :].cpu().numpy()

        if args.debug_mode:
            input("PRESS ENTER FOR NEXT STEP")

        viz.save("RGB", BEV_map.mapping_module.rgb_bev, step)
        if args.show_depth:
            viz.save("Depth", BEV_map.mapping_module.depth_bev, step)
        if args.show_dilated:
            viz.save("dilated occupancy map", graph.dilated_occupancy_map, step, cmap="gray",vflip=True)
        if args.show_frontiers:
            viz.save("frontier map", graph.frontier_map, step, cmap="gray",vflip=True)
        if args.show_map:
            viz.save("map | position | trajectory | goal", BEV_map.global_map_position_trajectory, step, vflip=True)
        if args.show_explore:
            viz.save("exploration map", BEV_map.global_exploration_map, step, cmap="gray",vflip=True)
        viz.refresh(step, title=f" Exploring and Searching | step {step}",block = False)

        obs, rgbd, done, infos = agent.step(agent_input)
        step += 1

if __name__ == "__main__":
    main()

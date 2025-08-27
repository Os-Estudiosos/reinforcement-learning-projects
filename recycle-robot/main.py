from src.recycicle_robot import *

def main():
    robot = RecycleRobot()
    robot.train()
    robot.save_rewards()
    robot.plot_best_policy()
    

if __name__ == "__main__":
    main()

from matplotlib import pyplot as plt
from mushroom_rl.core import Environment
import gymnasium as gym
from gymnasium.wrappers import OrderEnforcing
import time
import keyboard


# images: Start, Agent, Puddle, Ice, End
# start: fl.get_map()->
# step -> state = agent -> convert to map
# converter map to rendering picture


# Create the FrozenLake environment using Gym
mdp1 = Environment.make('Gym', 'FrozenLake-v1', is_slippery=False, render_mode="human")
mdp_img = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
#mdp = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
mdp = gym.make("CliffWalking-v0", render_mode="human")
mdp = OrderEnforcing(mdp, disable_render_order_enforcing=True)

print(mdp.reset())
print(mdp.step(2))
print(mdp.s)
# Create a dictionary to map keys to actions
key_to_action = {
    'w': 3,  # Move up
    'a': 0,  # Move left
    's': 1,  # Move down
    'd': 2  # Move right
}

def save_rendered_image(env, filename):
    """Function to save the rendered image to a file."""
    img = env.render()
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def manual_control():
    """Function to manually control the agent using the keyboard."""
    state = mdp.reset()  # Reset the environment
    done = False

    while not done:
        if keyboard.is_pressed('w'):  # Move up
            action = 3
        elif keyboard.is_pressed('a'):  # Move left
            action = 0
        elif keyboard.is_pressed('s'):  # Move down
            action = 1
        elif keyboard.is_pressed('d'):  # Move right
            action = 2
        else:
            action = None

        if action is not None:
            # Perform the action in the environment
            state, reward, done, info, _ = mdp.step(action)

            # Render the environment after each step
            mdp.render()
            #save_rendered_image(mdp, f'rendered_step_{time.time()}.png')
            time.sleep(0.5)  # Sleep for a short duration to control the speed of the game

        if done:
            print("Game Over!")
            break


# Run the manual control loop
manual_control()
mdp1.reset()
mdp1.step([2])
mdp1.step([1])


import copy
import gym
from matplotlib import pyplot as plt
from mushroom_rl.environments import Gym


class GridWorldWrapper:
    """
    Wrapper for grid world environments
    :param mdp: The environment to wrap
    :param depiction: The format in which to return the state representation.
    0: numeric, 1: coordinates, 2: map, 3: image
    """

    def __init__(self, mdp, depiction=0):
        self.mdp = mdp
        self.deception = depiction

    @property
    def current_state(self):
        """
        Return the current state of the grid world.
        Returns: Numeric representation of the current state

        """
        return self.mdp.env.unwrapped.s

    @property
    def shape(self):
        """
        Return the shape of the grid world depending on the environment.
        Returns: (rows, columns)
        """
        raise NotImplementedError

    def get_coordinates(self):
        """
        Get the coordinates of the current state in the grid world.
        Returns: (row, column)

        """
        rows, cols = self.shape
        row = self.current_state // cols
        colum = self.current_state % cols
        return row, colum

    def get_coordinates_for_state(self, state):
        """
        Get the coordinates of the given state in the grid world.
        """
        if state < 0 or state >= self.mdp.env.unwrapped.observation_space.n:
            raise ValueError("Invalid state")
        original_state = self.mdp.env.unwrapped.s
        self.mdp.env.unwrapped.s = state
        coordinates = self.get_coordinates()
        self.mdp.env.unwrapped.s = original_state
        return coordinates

    def get_map(self):
        """
        Get the map of the grid world with the current state marked.
        Returns: 2D list of the grid world

        """
        raise NotImplementedError

    def get_map_for_state(self, state):
        """
        Get the map of the grid world with the given state marked.
        """
        if state < 0 or state >= self.mdp.env.unwrapped.observation_space.n:
            raise ValueError("Invalid state")
        original_state = self.mdp.env.unwrapped.s
        self.mdp.env.unwrapped.s = state
        map = self.get_map()
        self.mdp.env.unwrapped.s = original_state
        return map

    def get_img(self):
        """
        Get the image of the grid world with the current state marked.
        Returns: 3D numpy array of the image

        """
        return self.mdp.env.unwrapped.render()

    def get_img_for_state(self, state):
        """
        Get the image of the grid world with the given state marked.
        """
        if state < 0 or state >= self.mdp.env.unwrapped.observation_space.n:
            raise ValueError("Invalid state")
        original_state = self.mdp.env.unwrapped.s
        self.mdp.env.unwrapped.s = state
        img = self.get_img()
        self.mdp.env.unwrapped.s = original_state
        return img

    def get_state(self):
        """
        Get the current state in the desired format.
        Returns: The current state in the desired format

        """
        if self.deception == 0:
            return self.current_state
        elif self.deception == 1:
            return self.get_coordinates()
        elif self.deception == 2:
            return self.get_map()
        elif self.deception == 3:
            return self.get_img()


class FrozenLakeWrapper(GridWorldWrapper):
    """
    Wrapper for the FrozenLake environment
    :param mdp: The environment to wrap
    :param depiction: The format in which to return the state representation.
    0: numeric, 1: coordinates, 2: map, 3: image
    """
    @property
    def shape(self):
        """
        Return the shape of the FrozenLake environment.
        """
        n_states= self.mdp.env.unwrapped.observation_space.n
        return int(n_states ** 0.5), int(n_states ** 0.5)

    def get_map(self):
        """
        Get the map of the FrozenLake environment with the current state marked:
        1. S: Start
        2. F: Frozen
        3. H: Hole
        4. G: Goal
        5. A: Agent
        6. D: Dead
        Returns: 2D list of the FrozenLake environment

        """
        cur_row, cur_colum = self.get_coordinates()
        map = copy.deepcopy(self.mdp.env.unwrapped.desc)
        if map[cur_row][cur_colum] == b'H':
            map[cur_row][cur_colum] = "D"
        else:
            map[cur_row][cur_colum] = "A"
        return map


class CliffWalkingWrapper(GridWorldWrapper):
    """
    Wrapper for the CliffWalking environment
    :param mdp: The environment to wrap
    :param depiction: The format in which to return the state representation.
    0: numeric, 1: coordinates, 2: map, 3: image
    """
    @property
    def shape(self):
        """
        Return the shape of the CliffWalking environment.
        """
        return self.mdp.env.unwrapped.shape

    def get_map(self):
        """
        Get the map of the CliffWalking environment with the current state marked:
        1. S: Start
        2. F: Grass
        2. C: Cliff
        3. G: Goal
        4. A: Agent
        Returns: 2D list of the CliffWalking environment

        """
        cur_row, cur_colum = self.get_coordinates()
        rows, cols = self.mdp.env.unwrapped.shape
        map = [['F' for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if r == rows - 1 and 0 < c < cols - 1:
                    map[r][c] = 'C'
                elif r == rows - 1 and c == 0:
                    map[r][c] = 'S'
                elif r == rows - 1 and c == cols - 1:
                    map[r][c] = 'G'
        map[cur_row][cur_colum] = 'A'
        return map


def get_all_formats_step(fl, num_step):
    print("Numeric:")
    print(fl.get_state())
    print("Coordinates:")
    print(fl.get_coordinates())
    print("Map:")
    print(fl.get_map())
    img = fl.get_img()
    save_path = "Render" + str(num_step) + ".png"
    plt.imsave(save_path, img)


mdp_f = Gym('FrozenLake-v1', is_slippery=False, gamma=.9, render_mode="rgb_array")
fl = FrozenLakeWrapper(mdp_f, depiction=0)
fl.mdp.reset()

"""
print(fl.get_state())
print(fl.get_coordinates())
fl.mdp.step([1])
print(fl.get_state())
img = fl.get_img_for_state(3)
plt.imshow(img)
plt.show()

mdp_c = Gym("CliffWalking-v0", render_mode="rgb_array", horizon=100)
cw = CliffWalkingWrapper(mdp_c, depiction=0)
cw.mdp.reset()
print(cw.get_state())
print(cw.get_coordinates())
print(cw.get_map())
img = cw.get_img_for_state(4)
plt.imshow(img)
plt.show()
#get_all_formats_step(cw, 0)
"""
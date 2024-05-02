"""Gym environment for pushing shape tasks (2D Shapes and 3D Cubes)."""

import logging

import numpy as np
# import matplotlib as mpl
# mpl.use("TkAgg")
# mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image, ImageOps
from skimage.draw import polygon, disk
from cv2 import circle

import gym
from gym import spaces

# from mbrl.seeding import np_random_seeding
# from grid import Grid

logger = logging.getLogger(__name__)


def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
    return polygon(rr, cc, im_size)


def triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width], [c0 + width // 2, c0, c0 + width]
    return polygon(rr, cc, im_size)


def fig2rgb_array(fig):
    fig.canvas.draw()
    buffer = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    return np.fromstring(buffer, dtype=np.uint8).reshape(height, width, 3)


def render_cubes(positions, width):
    voxels = np.zeros((width, width, width), dtype=np.bool)
    colors = np.empty(voxels.shape, dtype=object)

    cols = ["purple", "green", "orange", "blue", "brown"]

    for i, pos in enumerate(positions):
        voxels[pos[0], pos[1], 0] = True
        colors[pos[0], pos[1], 0] = cols[i]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 1.0))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.line.set_lw(0.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.voxels(voxels, facecolors=colors, edgecolor="k")

    im = fig2rgb_array(fig)
    plt.close(fig)
    im = np.asarray(Image.fromarray(im[215:455, 80:570]).resize((50, 50), Image.ANTIALIAS))  # Crop and resize
    return im / 255.0


class ShapeGridworld(gym.Env):
    """Gym environment for shape pushing task."""
    DEFAULT_COLOR = (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0)

    def __init__(
        self,
        width=5,
        height=None,
        x_step=1,                        # maximum steps in x translation actions
        y_step=None,                     # maximum steps in y translation actions
        render_type="circles",
        render_delta=10,
        num_objects=None,
        object_persistency=10,
        render_w_grid=False,
        seed=None,
        *,
        max_dist=-1,    # setting this to negative is the same as having no constraints in randomization of initial positions on reset
        control="all",  # 'limited' or 'all'
        control_boundaries=None,
        **kwargs,
    ):
        self.width = width
        self.height = height if height is not None else self.width
        self.render_delta = render_delta

        self.x_step = x_step if x_step >= 1 else self.width * x_step
        self.y_step = (y_step if y_step >= 1 else self.height * y_step) if y_step is not None else self.x_step

        self.render_type = render_type
        self.render_w_grid = render_w_grid
        # No grid can be printed due to the slanted view with cubes!
        if self.render_type == "cubes":
            self.render_w_grid = False

        # self.num_objects = num_objects
        self.object_persistency = object_persistency

        self.max_dist = max_dist
        self.control = control
        self.control_boundaries = control_boundaries

        self.np_random = None
        self.game = None
        self.target = None
        # If True, then check for collisions and don't allow two
        #   objects to occupy the same position.
        self.collisions = True

        self.seed(seed)

        self.num_actions = 2
        # self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_actions,), dtype=np.float32)

        # Initialize to pos outside of env for easier collision resolution.
        self.objects = [[-1, -1]] * (num_objects if num_objects else 0)

        # self.colors = utils.get_colors(num_colors=max(9, self.num_objects))
        # self.colors = [
        #     (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0) for _ in range(max(9, self.num_objects))
        # ]

        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(self.num_objects * 2 + 2,), dtype=np.float32
        # )

        # # Sets the mask for the objects that are allowed to move
        # self._mask = [1] * (num_objects if num_objects else 0)

        self.mask()
        self.checkpoint()
        self.reorder()
        self.reset()

    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, objects):
        self._objects = objects
        if not self.object_persistency and self.control == "all":
            self.num_actions = 2 * self.num_objects
        if not hasattr(self, 'colors') or len(self.colors) != self.num_objects:
            self.colors = [ShapeGridworld.DEFAULT_COLOR] * self.num_objects

    @property
    def num_objects(self):
        return len(self.objects)

    @property
    def observation_space(self):
        # TODO: `observation_space` is a mutable object. It should have a setter and be a member.
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_objects * 2 + (2 if self.object_persistency else 0),), dtype=np.float32)

    @property
    def num_actions(self):
        return self._num_actions

    @num_actions.setter
    def num_actions(self, num_actions):
        if not hasattr(self, 'num_actions') or self.num_actions != num_actions:
            self._num_actions = num_actions
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_actions,), dtype=np.float32)
            self.action_space.np_random = self.np_random

    @property
    def state_obs(self):
        if self.object_persistency:
            return np.hstack((np.asarray(self.objects, dtype=np.float32).flatten(), self.current_object, self.current_object_t))
        else:
            return np.asarray(self.objects, dtype=np.float32).flatten()

    @staticmethod
    def objects_from_state(state_obs, object_persistency=True):
        return state_obs[:-2 if object_persistency else None].reshape(-1, 2).tolist()

    def mask(self, mask=None):
        self._mask = np.asarray(self.mask_from_boundaries() if mask is None else mask, dtype=bool)
        if not self.object_persistency and self.control == "limited":
            self.num_actions = 2 * sum(self._mask)
        return self

    def mask_from_boundaries(self, boundaries=None):  # ((lower_row, lower_column), (upper_row, upper_column))
        if boundaries:
            self.control_boundaries = boundaries
        else:
            boundaries = self.control_boundaries

        if not boundaries:
            return [1] * self.num_objects

        if boundaries == "random":
            return np.random.choice([0, 1], self.num_objects, p=[0.5, 0.5])

        if isinstance(boundaries, float):
            return np.random.choice([0, 1], self.num_objects, p=[1 - boundaries, boundaries])

        def object_within_active_bounds(pos):
            if pos[0] < 0 or pos[1] < 0:
                return True

            lr = boundaries[0][0] if boundaries[0][0] >= 0 else 0
            ur = boundaries[1][0] if boundaries[1][0] >= 0 else self.height
            lc = boundaries[0][1] if boundaries[0][1] >= 0 else 0
            uc = boundaries[1][1] if boundaries[1][1] >= 0 else self.width

            return lr <= pos[0] < ur and lc <= pos[1] < uc

        return list(map(object_within_active_bounds, self.objects))

    def reorder(self):
        m = t = False
        for m in self._mask:
            if not m:
                t = True
            elif t:
                logger.debug(">> Readjusting object order ...")
                break

        if m and t and self.object_persistency:
            assert self.current_object == 0, f"Expected current_object to be 0 for reordering, got {self.current_object}."
            assert self.current_object_t == 0, f"Expected current_object_t to be 0 for reordering, got {self.current_object_t}."
        # current_object = self.objects[self.current_object]

        _reorder = lambda l: np.r_[np.asarray(l)[self._mask], np.asarray(l)[~self._mask]]

        self.objects = _reorder(self.objects).tolist()
        self.objects_ = _reorder(self.objects_).tolist()
        self.colors = _reorder(self.colors).tolist()
        self._mask = _reorder(self._mask)
        # self.current_object = self.objects.index(current_object)
        return self

    def seed(self, seed=None):
        self.np_random, seed = np.random.default_rng(seed)
        # self.action_space.np_random = self.np_random
        self._seed = seed
        return [seed]

    def render_image(self, *, format='image', highlight_active=False, invert_grid=False, **kwargs):  # color=True
        image = self.render(**kwargs)  # (self.render(**kwargs) * 255).astype(np.uint8)
        if invert_grid:
            image = 255 - image
        if highlight_active:
            # image = self.highlight_active(image, v=[int(invert_grid) * 255] * (3 if image.mode == 'RGB' else 1) if image.mode == 'L' else [int(not invert_grid) * 255])
            image = self.highlight_active(image, v=[255, 0, 0] if image.ndim == 3 else [int(invert_grid) * 255])
        if format == 'image':
            return Image.fromarray(image)
        return image
    image = property(render_image)

    def render(self, **kwargs):  # figsize=(4, 4), dpi=56, fast=True
        im = self._render_callback(**kwargs)
        # im = np.transpose(im, (1, 2, 0))

        if self.render_w_grid:
            dx, dy = self.render_delta, self.render_delta
            # Custom (rgb) grid color
            grid_color = [1, 1, 1]
            # Modify the image to include the grid
            im[:, ::dy, :] = grid_color
            im[::dx, :, :] = grid_color

        return im

    def _render_callback(self, **kwargs):  # color=True
        if self.render_type == "grid":
            im = np.zeros((self.width, self.height, 3), dtype=np.uint8)
            for idx, pos in enumerate(self.objects):
                im[(*pos,)] = self.colors[idx][:3]
            return im  # .transpose(im, (1, 2, 0))
        elif self.render_type == "circles":
            # im = np.zeros((self.width * self.render_delta, self.height * self.render_delta, 3), dtype=np.float32)
            im = np.full((self.width * self.render_delta, self.height * self.render_delta, 3), 0, dtype=np.uint8)
            for idx, pos in enumerate(self.objects):
                if pos[0] < 0 or pos[1] < 0:
                    logger.debug(f"> Skipping rendering object {idx} at {pos}.")
                    continue
                # rr, cc = disk(
                #     (
                #         pos[0] * self.render_delta + self.render_delta / 2,
                #         pos[1] * self.render_delta + self.render_delta / 2,
                #     ),
                #     self.render_delta / 2,
                #     shape=im.shape,
                # )
                # im[rr, cc, :] = self.colors[idx][:3]
                circle(
                    im,
                    (
                        int(pos[1] * self.render_delta + self.render_delta / 2),
                        int(pos[0] * self.render_delta + self.render_delta / 2),
                    ),
                    int(self.render_delta / 2),
                    color=self.colors[idx][:3],
                    thickness=-1,
                )
            return im  # .transpose([2, 0, 1])
        elif self.render_type == "shapes":
            # im = np.zeros((self.width * self.render_delta, self.height * self.render_delta, 3), dtype=np.float32)
            im = np.full((self.width * self.render_delta, self.height * self.render_delta, 3), 0, dtype=np.uint8)
            for idx, pos in enumerate(self.objects):
                if idx % 3 == 0:
                    rr, cc = disk(
                        (
                            pos[0] * self.render_delta + self.render_delta / 2,
                            pos[1] * self.render_delta + self.render_delta / 2,
                        ),
                        self.render_delta / 2,
                        shape=im.shape,
                    )
                    im[rr, cc, :] = self.colors[idx][:3]
                    # circle(
                    #     im,
                    #     (
                    #         int(pos[1] * self.render_delta + self.render_delta / 2),
                    #         int(pos[0] * self.render_delta + self.render_delta / 2),
                    #     ),
                    #     int(self.render_delta / 2),
                    #     color=self.colors[idx][:3],
                    #     thickness=-1,
                    # )
                elif idx % 3 == 1:
                    rr, cc = triangle(
                        pos[0] * self.render_delta, pos[1] * self.render_delta, self.render_delta, im.shape
                    )
                    im[rr, cc, :] = self.colors[idx][:3]
                else:
                    rr, cc = square(pos[0] * self.render_delta, pos[1] * self.render_delta, self.render_delta, im.shape)
                    im[rr, cc, :] = self.colors[idx][:3]
            return im  # .transpose([2, 0, 1])
        elif self.render_type == "cubes":
            im = render_cubes(self.objects, self.width)
            return im  # .transpose([2, 0, 1])

    def _ipython_display_(self):
        from IPython.display import display
        display(self.image)  # noqa: F821

    def highlight_active(self, image: np.ndarray, v: list):
        if self.render_type == 'circles':
            from mbrl.environments.imagelib import Im
            if self.object_persistency:
                image = Im.highlight(image, *self.objects[self.current_object], size=self.render_delta, v=v)
            elif self.control == 'limited':
                for (r, c), m in zip(self.objects, self._mask):
                    if m:
                        image = Im.highlight(image, r, c, size=self.render_delta, v=v)
            return image
        else:
            raise NotImplementedError(f"Highlighting not implemented for render_type '{self.render_type}'.")

    def checkpoint(self, objects=None):
        self.objects_ = [o[:] for o in (objects if objects else self.objects)]  # Original positions for resetting
        return self

    def restore(self):
        self.objects = [o[:] for o in self.objects_]
        return self

    def reset(self, *, constrained=True):
        # self.objects = [[-1, -1]] * (self.num_objects if self.num_objects else 0)

        if self.objects and not self.objects_:
            logger.warning("> No checkpoint found! | Resetting objects in boundaries without constraints and leaving objects outside boundaries unchanged.")
            self.objects_ = [[-1, -1] if m else self.objects[i] for i, m in enumerate(self._mask)]
        else:
            assert self.num_objects == len(self.objects_), f"Checkpoint has {len(self.objects_)} objects, but state has {self.num_objects}."

        for i in range(self.num_objects):
            if self._mask[i]:
                self.objects[i] = [-1, -1]
            else:  # if self.control == "all"  # Only required when all objects are allowed to move
                self.objects[i] = self.objects_[i][:]

        # Randomize object positions
        for i in range(self.num_objects):
            # Resample to ensure objects don't fall on same spot.
            if self._mask[i]:
                while not self.valid_pos(self.objects[i], i, constrained=constrained):
                    self.objects[i] = [
                        self.np_random.choice(np.arange(self.width)),
                        self.np_random.choice(np.arange(self.height)),
                    ]

        if self.object_persistency:
            self.current_object = 0
            self.current_object_t = 0

        return self.state_obs

    def valid_pos(self, pos, obj_id, *, constrained=False):
        """Check if position is valid."""
        if pos[0] < 0 or pos[0] >= self.width:
            return False
        if pos[1] < 0 or pos[1] >= self.height:
            return False

        if constrained and self.max_dist >= 0:  # self._mask[obj_id]  # Only required for unlocked objects
            if self.objects_[obj_id][0] >= 0 and self.objects_[obj_id][1] >= 0 and abs(pos[0] - self.objects_[obj_id][0]) + abs(pos[1] - self.objects_[obj_id][1]) > self.max_dist:
                return False

        if self.collisions:
            for idx, obj_pos in enumerate(self.objects):
                if idx == obj_id:
                    continue

                if pos[0] == obj_pos[0] and pos[1] == obj_pos[1]:
                    return False

        return True

    def valid_move(self, obj_id, offset, *, constrained=False):
        """Check if move is valid."""
        if self.control == "limited" and not self._mask[obj_id] and offset != [0, 0]:
            logger.error(f"> Tried moving locked object, id={obj_id} ({self.objects[obj_id]})!")
            # return False
        old_pos = self.objects[obj_id]
        new_pos = [p + o for p, o in zip(old_pos, offset)]
        return self.valid_pos(new_pos, obj_id, constrained=constrained)

    def translate(self, obj_id, offset):
        """ "Translate object pixel.

        Args:
            obj_id: ID of object.
            offset: (x, y) tuple of offsets.
        """

        if self.valid_move(obj_id, offset):
            self.objects[obj_id][0] += offset[0]
            self.objects[obj_id][1] += offset[1]

    def discretize_action(self, action, step=1):
        return (action + 1) * (step + 0.5) // 1 - step // 1

    def step(self, action):
        done = False
        reward = 0

        # directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        direction = [self.discretize_action(a, s) for a, s in zip(action, [self.x_step, self.y_step] * (len(action) // 2))]

        if self.object_persistency:
            obj = self.current_object
            # logger.debug(f"Stepping for object {obj} in direction {direction}")

            self.translate(obj, direction)

            self.current_object_t += 1

            if self.current_object_t >= self.object_persistency:
                self.current_object = (self.current_object + 1) % self.num_objects
                if self.control == "limited":
                    while not self._mask[self.current_object]:
                        self.current_object = (self.current_object + 1) % self.num_objects
                self.current_object_t = 0
        else:
            assert len(direction) == self.num_actions, f"Expected {self.num_actions} actions, got {len(direction)}."
            for i, dir in enumerate(zip(direction[::2], direction[1::2])):
                self.translate(i, dir)

        return self.state_obs, reward, done, None

    def regularity(self, **kwargs):
        from mbrl.environments.imagelib import Im
        return Im.regularity(self.objects, **kwargs)

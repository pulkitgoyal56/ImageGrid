#!/usr/bin/env python3
# coding: utf-8

from fire import Fire
import logging
from itertools import product
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from skimage.morphology import skeletonize
# from skimage.draw import disk
from cv2 import circle

from shape_gridworld import ShapeGridworld


logger = logging.getLogger(__name__)

logging.getLogger('PIL').setLevel(logging.INFO)


class Im:
    def __init__(self, basename='', category='', prefix='', *, base_dir='', **kwargs):
        self.base_dir = Path(base_dir)
        self.category = category

        self.path = self.base_dir / self.category / prefix / basename

        self.load(**kwargs)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        if path:
            self._path = Path(path)

    @property
    def dirname(self):
        return self.path.parent

    @property
    def basename(self):
        return Path(self.path.name)

    @dirname.setter
    def dirname(self, dirname):
        if dirname is not None:
            self.path = dirname / self.basename

    @basename.setter
    def basename(self, basename):
        if basename:
            self.path = self.dirname / basename

    def load(self, image=None, basename='', dirname=None, *, create=None, mode='1', invert=False, skeleton=False, overwrite=True, **kwargs):
        self.dirname = dirname
        self.basename = basename

        if mode:
            mode = str(mode)
            if skeleton and mode == '1':
                logger.warning("`skimage.morphology.skeletonize()` crashes with segmentation fault for binary images. Using grayscale instead.")
                mode = 'L'

        if create and not image:
            self.image = Im.create_image(create, mode=mode, **kwargs)
        elif image or self.path.is_file() and self.path.suffix == ".png":
            try:
                if image:
                    self.image = image.copy()
                else:
                    logger.debug(f"> Loading image @ file://{self.path.resolve()}.")
                    self.image = Image.open(self.path)
            except UnidentifiedImageError:
                if overwrite:
                    logger.error(f">> Image @ file://{self.path.resolve()} is corrupted; falling back to random image.")
                    self.image = Im.create_image('random', mode=mode, **kwargs)
                else:
                    raise ValueError(f"> Image @ file://{self.path.resolve()} is corrupted.")
            else:
                if mode:
                    logger.debug(f">> Converting image to { {'1': 'binary', 'L': 'grayscale'}.get(mode, f'`{mode}`') }.")
                    self.image = self.image.convert(mode)
        elif overwrite:
            logger.error(f"> No image @ file://{self.path.resolve()}, and no image or creation method provided; falling back to random image.")  # empty image
            self.image = Im.create_image('random', mode=mode, **kwargs)  # 'empty'
        else:
            raise FileNotFoundError(f"> No image @ file://{self.path.resolve()}, and no image or creation method provided.")

        if invert:
            logger.debug(">> Inverting image.")
            self.apply(ImageOps.invert)

        if skeleton:
            logger.debug(">> Using image skeleton.")
            if mode and mode not in ('L', 'RGB'):
                logger.warning(">>> `skimage.morphology.skeletonize()` doesn't work well for image modes other than 'L' and 'RGB'.")
                # logger.debug(">>>> Converting image to grayscale before skeletonizing.")
                # self.image.convert('L')
            self.apply(skeletonize, array=True)

        if overwrite:
            if self.path.is_file() and (image or create):
                self.save(grid_image=False)
            elif basename and (skeleton or invert or mode):
                self.save(grid_image=False, suffix='_pp')

        return self

    def save(self, prefix='', suffix='', *, grid_image=True, image=None, **kwargs):
        """Overwrites file, if exists."""
        if not prefix and grid_image:
            prefix = f"pre-processed_{self.grid.width}_{self.grid.height}_{self.grid.render_delta}_?_{self.grid.render_w_grid}_{self.grid.render_type}"
        filepath = self.dirname / prefix / (self.basename.stem + suffix + (self.basename.suffix if self.basename.suffix else ".png"))
        filepath.parent.mkdir(parents=True, exist_ok=True)
        image = self.get_image(grid_image=grid_image, **kwargs) if image is None else image
        image.save(filepath)
        logger.debug(f"> {'Grid image' if grid_image else 'Image'} saved @ file://{filepath.resolve()}.")
        return image

    def apply(self, func, *, array=False, inplace=True, **kwargs):
        image = Image.fromarray(func(np.asarray(self.image), **kwargs)) if array else func(self.image, **kwargs)
        if not inplace:
            return image
        self.image = image
        return self

    def register(self, grid_width=28, grid_height=28, gridcell_size=8, *, threshold_ratio=0, **kwargs):
        self.grid = ShapeGridworld(grid_height, grid_width, render_delta=gridcell_size, **kwargs)
        self.grid.objects = []
        self.grid.colors = []

        image = np.asarray(self.image.resize((self.grid.height * self.grid.render_delta, self.grid.width * self.grid.render_delta), Image.BICUBIC), dtype=int)
        image = np.atleast_3d(image).mean(axis=-1)  # image.reshape(self.grid.height * self.grid.render_delta, self.grid.width * self.grid.render_delta, -1).mean(axis=-1)
        kernel = Im.create_kernel(self.grid.render_delta, self.grid.render_type)
        max_value = np.sum(kernel) * (1 if self.image.mode == '1' else 255)
        min_value = threshold_ratio * max_value

        for i, j in product(range(self.grid.width), range(self.grid.height)):
            if (value := np.tensordot(
                image[
                    self.grid.render_delta * i : self.grid.render_delta * (i + 1),
                    self.grid.render_delta * j : self.grid.render_delta * (j + 1)
                ],
                kernel
            )) > min_value:
                self.grid.objects.append([i, j])
                # self.grid.colors.append(([value / max_value] * 3 + [1.]) if self.image.mode != '1' else (1., 1., 1., 1.))  # self.grid.DEFAULT_COLOR
                self.grid.colors.append(([int(value / max_value * 255)] * 3 + [255]) if self.image.mode != '1' else (255, 255, 255, 255))  # self.grid.DEFAULT_COLOR

        self.grid.mask()
        self.grid.checkpoint()
        self.grid.reorder()

        logger.debug(f"> {self.grid.num_objects}/{self.grid.width * self.grid.height} objects registered.")

        return self.grid

    def get_grid_image(self, **kwargs):
        return self.grid.render_image(**kwargs)
    grid_image = property(get_grid_image)

    def get_image(self, *, grid_image=False, invert=False, **kwargs):
        if grid_image:
            return self.get_grid_image(invert_grid=invert, **kwargs)
        else:
            image = self.image
            return ImageOps.invert(image) if invert else image

    def deform(self):
        if self.grid.objects:
            action = self.grid.action_space.sample()
            return self.grid.step(action), action
        else:
            logger.warn("> No objects to deform.")

    def show(self, axes=None, figure=None, figsize=(5, 5), title='', cmap="gray", *, grid_image=False, invert=False, **kwargs):
        show = False
        if axes is None:
            if figure is None:
                figure, axes = plt.subplots(1, 1, figsize=figsize)
                show = True
            else:
                axes = figure.gca()
        elif figure is None:
            figure = axes.get_figure()

        axes.axis("off")
        axes.imshow(self.get_image(grid_image=grid_image, invert=invert, **kwargs), vmin=0, vmax=256 if cmap == 'gray' else 1, cmap=cmap, axes=axes)
        axes.set_title(title)

        if show:
            logger.debug(f"> Showing {'*colored* ' if self.image.mode == 'RGB' else '*grayscale* ' if self.image.mode == 'L' else ''}{'*inverted* ' if invert else ''}{'grid image' if grid_image else 'image'}.")
            plt.show()
            return figure, axes

    def _ipython_display_(self):
        if self.image:
            from IPython.display import display
            display(self.image)  # noqa: F821

    def __call__(self, show=True, *args, **kwargs):
        if show:
            self.show(*args, **kwargs)
        else:
            return self.get_image(**kwargs)

    @staticmethod
    def create_image(method='random', **kwargs):
        if method == 'random':
            return Im.create_random_image(**kwargs)
        elif method == 'empty':
            return Im.create_empty_image(**kwargs)
        elif method:
            raise ValueError(f"> Unsupported image creation method {method}.")
        else:
            raise ValueError("> No image creation method set (set fallback='random').")

    @staticmethod
    def create_random_image(width=28, height=28, mode='1', *, p=0.5, seed=None, **kwargs):  # mode: '1' | 'L'
        logger.debug(f"> Generating random image. | {width} x {height} ; {mode=} | {p=} @ {seed=}")
        if seed:
            np.random.seed(seed)
        if str(mode) == '1':
            return Image.fromarray(np.random.choice([False, True], size=(int(height), int(width)), p=[1 - float(p), float(p)]))
        elif mode == 'L':
            return Image.fromarray(np.random.binomial(256, p, size=(int(height), int(width))).astype(np.uint8))
        elif mode == 'RGB':
            return Image.fromarray(np.rollaxis(np.random.binomial(256, p, size=(int(height), int(width))).astype(np.uint8).T, 1))
        else:
            raise ValueError(f"> Convert mode `{mode}` not supported.")

    @staticmethod
    def create_empty_image(width=28, height=28, mode='1', **kwargs):  # mode: '1' | 'L'
        logger.debug(f"> Using empty (black) image. | {width} x {height} ; {mode=}")
        if str(mode) == '1':
            return Image.fromarray(np.zeros((int(height), int(width)), dtype=bool))
        elif mode == 'L':
            return Image.fromarray(np.zeros((int(height), int(width)), dtype=np.uint8))
        elif mode == 'RGB':
            return Image.fromarray(np.zeros((int(height), int(width), 3), dtype=np.uint8))
        else:
            raise ValueError(f"> Convert mode `{mode}` not supported.")

    @staticmethod
    def create_kernel(gridcell_size, shape="circles"):
        if shape == "grid" or gridcell_size in [1, 2]:
            return np.ones((gridcell_size, gridcell_size), dtype=bool)
        elif shape == "circles":
            radius = gridcell_size / 2
            center = (int(radius),) * 2

            y, x = np.ogrid[:gridcell_size, :gridcell_size]
            dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

            return dist_from_center <= radius

    @staticmethod
    def create_mask(size, pixel_size):
        image = Image.new("1", size, "black")
        width, height = size
        image_draw = ImageDraw.Draw(image)
        for x in range(0, width - pixel_size + 1, pixel_size):
            for y in range(0, height - pixel_size + 1, pixel_size):
                image_draw.ellipse((x, y, x + pixel_size, y + pixel_size), fill="white", outline="white")
        return image

    @staticmethod
    def gridcell_value(image, x, y, gridcell_size, shape="squares"):
        image = np.asarray(image)
        if shape == "squares":
            return image[
                x - int(gridcell_size / 2) : x + int(gridcell_size / 2),
                y - int(gridcell_size / 2) : y + int(gridcell_size / 2)
            ].any()
        elif shape == "circles":
            for (i, j), v in np.ndenumerate(
                image[
                    x - int(gridcell_size / 2) : x + int(gridcell_size / 2),
                    y - int(gridcell_size / 2) : y + int(gridcell_size / 2)
                ]
            ):
                if (
                    np.sqrt((i - gridcell_size / 2) ** 2 + (j - gridcell_size / 2) ** 2) < (gridcell_size / 2) ** 2
                ) and v:
                    return True

    @staticmethod
    def mask_grid(image, gridcell_size, method="mask"):  # 'grid'
        if method == "mask":
            return Image.fromarray(np.asarray(Im.create_mask(image.size, gridcell_size)) * np.asarray(image))
        elif method == "grid":
            grid_image = np.zeros(image.size, dtype=bool)
            for x, y in zip(
                *map(
                    np.ravel,
                    np.mgrid[0 : image.size[0] : int(gridcell_size / 2), 0 : image.size[1] : int(gridcell_size / 2)]
                )
            ):
                grid_image[
                    x - int(gridcell_size / 2) : x + int(gridcell_size / 2),
                    y - int(gridcell_size / 2) : y + int(gridcell_size / 2)
                ] = Im.gridcell_value(image, x, y, gridcell_size)
            return Image.fromarray(grid_image)

    @staticmethod
    def pixelate(image, pixel_size):
        return (
            image.resize(tuple(map(lambda s: int(s / pixel_size), image.size)), resample=Image.Resampling.BILINEAR)
            .convert("1")
            .resize(image.size, Image.Resampling.NEAREST)
        )

    @staticmethod
    def shift_and_scale(image):
        pass

    @staticmethod
    def highlight(image: np.ndarray, r: int, c: int, size: int, v: list, *, radius: int = None):
        # image = np.asarray(image).copy()
        # image[(*disk((r * size + size / 2, c * size + size / 2), radius if radius else size / 4, shape=image.shape),)] = v
        circle(image, (int(c * size + size / 2), int(r * size + size / 2)), int(radius if radius else size / 4), color=v, thickness=-1)
        return image  # Image.fromarray(image)

    @staticmethod
    def show_images(filenames, category, prefix='', *, base_dir='', probabilities=None, mode=None, invert=False, mask=None, axes=None, figure=None, figsize=None):
        if not len(filenames):
            logger.warn("> No files to show.")
            return

        show = False
        if axes is None:
            if figure is None:
                figure, axes = plt.subplots(
                    int(np.ceil(len(filenames) / 10)), 10,
                    figsize=figsize if figsize else (15, int(np.ceil(len(filenames) / 10)) * 2),
                    sharex=True,
                    sharey=True,
                )
                figure.set_facecolor("white" if invert else "black")
                show = True
            else:
                raise ValueError("> `axes` must be provided if `figure` is provided.")
        else:
            for ax in axes.ravel():
                ax.set_facecolor("white" if invert else "black")

        normalize = np.log(len(probabilities[filenames[0]]))  # 1.
        axes = iter(axes.ravel())
        for filename in filenames:
            ax = next(axes)
            ax.imshow(Im(f"{filename}.png", category, prefix, base_dir=base_dir, mode=mode).image, cmap="gray")
            if probabilities:
                ax.set_title(
                    rf"$\bf{{{probabilities[filename][0]:.2f}}}$ | {-probabilities[filename] @ np.log(probabilities[filename]) / normalize:.2f}",
                    color="white" if not mask else "green" if mask[filename] else "red"
                )
            ax.axis("off")
        for ax in axes:
            ax.axis("off")
            ax.margins(0, 0)

        if show:
            plt.tight_layout()
            plt.show()
            return figure, axes

    @staticmethod
    def regularity(objects, metric=lambda a, b: a - b, cost=True, shift=False, *, normalize=False, precision=1., granularity=1., bidirectional=False, **kwargs):
        multiset = Counter(map(tuple, (np.floor(((lambda x: x) if bidirectional else np.abs)(metric(np.expand_dims(objects, 1), objects)[((lambda x: x) if bidirectional else np.triu)(~np.eye((n := len(objects)), dtype=bool))]) * precision / granularity) * granularity).astype(int)))
        distribution = np.asarray(list(multiset.values()))
        assert sum(distribution) == (nt := n * (n - 1) / (1 if bidirectional else 2))
        r = (distribution @ np.log(distribution) / nt - (0 if shift else np.log(nt))) * (-1 if cost else 1)
        return (r / np.log(nt), 1) if normalize else (r, np.log(nt))  # (regularity, max regularity)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    plt.set_loglevel("error")
    logging.getLogger('asyncio').setLevel(logging.ERROR)

    def main(*args, grid_image=False, invert_grid=False, **kwargs):
        image = Im(*args, **kwargs)
        if grid_image:
            image.register(*args, **kwargs)
        image.show(grid_image=grid_image, invert=invert_grid)

    Fire(main)

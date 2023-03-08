import random
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from typing import Generator, List, Tuple
from .utils import load_tile
from .tile_list import TileList
from .global_variables import CLASS_LABEL_TO_INDEX_MAP, NUM_CLASSES


class TileListBatchIterator: 
    """
    An iterator class to iterate over all tiles contained in a TileList object during network inference. 

    Attributes
    ----------
    _tile_list: TileList
        TileList object containing all relevant tiles.
    _tiles_dir: str
        Directory where the tiles are stored. 
    _tile_size: int
        Size of the (rectangular) tiles in pixels.
    _required_pixel_spacing: float
        Required pixel spacing in mm/px.
    _batch_size: int 
        Number of tiles to be in one batch.
    """

    def __init__(self, tile_list: TileList, tiles_dir: str,
                 tile_size: int, required_pixel_spacing: float, batch_size: int):
        """
        Constructor of TileListBatchIterator.    

        Parameters
        ----------
        tile_list: TileList
            TileList object containing all relevant tiles.
        tiles_dir: str
            Directory where the tiles are stored. 
        tile_size: int
            Size of the (rectangular) tiles in pixels.
        required_pixel_spacing: float
            Required pixel spacing in mm/px.
        batch_size: int 
            Number of tiles to be in one batch.
        """
        self._tile_list = tile_list
        self._tiles_dir = tiles_dir
        self._tile_size = tile_size
        self._required_pixel_spacing = required_pixel_spacing
        self._batch_size = batch_size

    @staticmethod
    def _scale_tile(tile: Image.Image) -> np.ndarray:
        """
        Scales image values to [-1, 1], the expected input for InceptionV3 network

        Parameters
        ---------- 
        tile: Image.Image
            Tile to be rescaled.

        Returns
        -------
        np.ndarray
            Tile with rescaled values. 
        """ 
        return (img_to_array(tile) / 127.5) - 1.0
    
    @staticmethod
    def _augment(tile: Image.Image) -> Image.Image: 
        """
        Performs simple data augmentation by random rotation of the tile.

        Parameters
        ---------- 
        tile: Image.Image
            Tile to be augmented.

        Returns
        -------
        np.ndarray
            Augmented tile. 
        """ 
        rotation_angle = random.choice([90, 180, 270, 360])
        return tile.rotate(angle=rotation_angle) 
    
    def __iter__(self):
        self._tile_index = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, List[Tuple[str, Tuple[int,int]]]]:
        """
        Prepares next batch of tiles. 

        Returns
        -------
        tuple
            Tuple with the first element being all tiles in the batch as np.ndarray and 
            the second element being a list of tile information corresponding to the tiles.
        """
        batch_images = np.empty((self._batch_size, self._tile_size, self._tile_size, 3))
        batch_tile_infos = [None] * self._batch_size
        curr_batch_size = 0

        while self._tile_index < len(self._tile_list) and curr_batch_size < self._batch_size:
            tile_info = self._tile_list.get_tile_info(self._tile_index)

            # Open and prepare tile
            tile = load_tile(self._tiles_dir, tile_info)
            tile = self._augment(tile) 
            tile = self._scale_tile(tile)
            
            # Add tile to batch
            batch_images[curr_batch_size] = tile[np.newaxis, ...] # add batch dimension
            batch_tile_infos[curr_batch_size] = tile_info
            
            curr_batch_size += 1
            self._tile_index += 1

        if curr_batch_size > 0:
              batch_images.resize((curr_batch_size, self._tile_size, self._tile_size, 3))
              batch_tile_infos = batch_tile_infos[0:curr_batch_size]
              return (batch_images, batch_tile_infos)
        else:
              raise StopIteration


def get_tile_generator(tile_list: TileList, tiles_dir: str, tile_size: int, 
                    required_pixel_spacing: float, batch_size: int, num_classes: int,
                    slides_metadata: pd.DataFrame, 
                    shuffle: bool=True) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:  
    """
    Yields batches for neural network training / validation.    

    Parameters
    ----------
    tile_list: TileList
        TileList object containing all relevant tiles.
    tiles_dir: str
        Directory where the tiles are stored. 
    tile_size: int
        Size of the (rectangular) tiles in pixels.
    required_pixel_spacing: float
        Required pixel spacing in mm/px. 
    batch_size: int
        Number of tiles to be in one batch.
    num_classes: int
        Number of classes in the current classification problem.
    slides_metadata: pd.DataFrame
        Metadata of the slides that were used in the prediction.
    shuffle: bool
        Bool indicating whether to shuffle tiles. 

    Returns 
    -------
    Generator
        Generator providing batches for training / validation. 
    """ 
    while True:
        if shuffle:
            tile_list.shuffle()

        batch_iterator = TileListBatchIterator(tile_list, tiles_dir, tile_size,
                                               required_pixel_spacing, batch_size)
        for batch_x, batch_tile_infos in batch_iterator:
            batch_y = np.empty((len(batch_tile_infos), num_classes))
            for i, (image_id, _ ) in enumerate(batch_tile_infos):
                slide_metadata = slides_metadata.loc[image_id]
                reference_value = CLASS_LABEL_TO_INDEX_MAP[slide_metadata['reference_class_label']]
                batch_y[i] = to_categorical(reference_value, num_classes)
            yield (batch_x, batch_y)
import os
from PIL import Image
from tqdm import tqdm
from typing import Callable 
from .utils import get_tile_path
from .wsi import WSIOpener
from .tile_list import TileList


def extract_tiles_if_not_existent(wsi_opener: WSIOpener, tiles_dir: str, tile_size: int, required_pixel_spacing: float, 
                     accept_function: Callable[[Image.Image], bool]) -> TileList:
    """
    Helper function that checks if extract_tiles (see there for more information) needs to be invoked.    
    """ 
    foreground_tiles_path = os.path.join(tiles_dir, 'foreground_tile_list.csv')
    
    if not os.path.exists(foreground_tiles_path):
        tile_list = extract_tiles(wsi_opener, tiles_dir, tile_size, required_pixel_spacing, accept_function) 
        tile_list.save(foreground_tiles_path)
    return TileList.load(foreground_tiles_path)

def extract_tiles(wsi_opener: WSIOpener, tiles_dir: str, tile_size: int, required_pixel_spacing: float, 
                     accept_function: Callable[[Image.Image], bool]) -> TileList:
    """
    Function that passes through a set of WSI and extracts tiles at a required pixel spacing 
    - if they are not already existing.    

    Parameters
    ----------
    wsi_opener: WSIOpener
        Instance of WSIOpener managing the required set of WSI.
    tiles_dir: str
        Directory where the tiles are stored. 
    tile_size: int
        Size of the (rectangular) tiles in pixels.
    required_pixel_spacing: float
        Required pixel spacing in mm/px. 
    accept_function: Callable
        Callable defining when a tile is accepted or rejected.

    Returns 
    -------
    TileList
        TileList instance comprising all extracted tiles. 
    """
    image_ids = wsi_opener.get_image_ids()
    tile_infos = []

    for image_id in tqdm(image_ids, total=len(image_ids)):

        wsi = wsi_opener.open_image(image_id)
        image_size = wsi.get_size_at_pixel_spacing(required_pixel_spacing)
        cols, rows = image_size[0] // tile_size, image_size[1] // tile_size
        for tile_pos_y in range(0, rows):
            for tile_pos_x in range(0, cols):
                tile_pos = (tile_pos_x, tile_pos_y)
                tile = wsi.get_tile(tile_pos, tile_size, required_pixel_spacing)
                assert tile.size[0] == tile.size[1] == tile_size
                if accept_function(tile):
                    tile_dir, tile_path = get_tile_path(tiles_dir, (image_id, tile_pos))
                    if not os.path.exists(tile_dir): 
                        os.makedirs(tile_dir)
                    if not os.path.exists(tile_path):                    
                        tile.save(tile_path, compress_level=1, optimize=False) 
                    tile_infos.append((image_id, tile_pos))
    
    return TileList(tile_infos)
import random
import pandas as pd
from typing import List, Tuple


class TileList:
    """
    Class representing a certain set of tiles.

    Attributes
    ----------
    _tile_infos: list
        List of information (i.e., corresponding image ID and position within the image) per tile. 
    """

    def __init__(self, tile_infos: List[Tuple[str, Tuple[int,int]]]):
        """
        Constructor of TileList.    

        Parameters
        ----------
        tile_infos: list
            List of information (i.e., corresponding image ID and position within the image) per tile. 
        """
        self._tile_infos = tile_infos

    def __len__(self) -> int: 
        return len(self._tile_infos)

    def get_tile_info(self, idx: int) -> Tuple[str, Tuple[int,int]]: 
        """
        Returns tile information, i.e. corresponding image ID and tile position.    

        Parameters
        ----------
        idx: int
            Index of tile of interest in the list of tiles.
        
        Returns 
        -------
        tuple
            Image ID and position within the image. 
        """
        return self._tile_infos[idx]

    def get_num_tiles(self) -> int:
        """
        Returns number of tiles in the TileList object.    
        
        Returns 
        -------
        int
            Number of tiles. 
        """
        return len(self._tile_infos)

    def get_num_tiles_of_image(self, image_id: str) -> int: 
        """
        Returns number of tiles belonging to a certain image.    

        Parameters
        ----------
        image_id: str
            Image ID of interest.
        
        Returns 
        -------
        int
            Number of tiles.
        """
        return sum(1 for tile_info in self._tile_infos if tile_info[0] == image_id)

    def get_random_tile_info(self) -> Tuple[str, Tuple[int,int]]:
        """
        Returns corresponding image ID and tile position of a random tile in 
        the TileList object.    
        
        Returns 
        -------
        tuple
            Image ID and tile position of the randomly selected tile.
        """
        return random.choice(self._tile_infos)

    def subset(self, image_ids: List[str]) -> 'TileList':
        """
        Creates a subset of the TileList object.    

        Parameters
        ----------
        image_ids: list
            List of image IDs, whose tiles should be present in the subset. 
        
        Returns 
        -------
        TileList
            TileList instance comprising all tiles belonging to the requested image IDs. 
        """
        subset_tile_infos = [tile_info for tile_info in self._tile_infos if tile_info[0] in image_ids]
        return TileList(subset_tile_infos)
    
    def shuffle(self) -> None: 
        """
        Randomly shuffles the list of tiles.     
        """
        random.shuffle(self._tile_infos)
    
    @classmethod
    def load(cls, path: str) -> 'TileList': 
        """
        Loads a TileList object from disk.    

        Parameters
        ----------
        path: str
            Full path to a csv file in the resective format.
        
        Returns 
        -------
        TileList
            TileList instance as loaded from disk. 
        """
        df = pd.read_csv(path)
        tile_infos = [(row['image_id'], (row['tile_position_x'], row['tile_position_y'])) 
                      for _i, row in df.iterrows()]
        return cls(tile_infos)
        
    def save(self, path: str) -> None: 
        """
        Saves the TileList object to disk.    

        Parameters
        ----------
        path: str
            Full path where to store the TileList object.
        """
        df = pd.DataFrame([(i, *j) for i,j in self._tile_infos], 
                          columns=['image_id', 'tile_position_x', 'tile_position_y'])
        df.to_csv(path, index=False)
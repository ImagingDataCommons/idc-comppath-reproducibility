import os 
import subprocess
import pandas as pd
from PIL import Image
from wsidicom import WsiDicom
from typing import List, Tuple


class WSILevel:
    """
    A class representing one level of a whole-slide image.

    Attributes
    ----------
    _dcm_image: 'WsiDicom'
        WsiDicom object holding one DICOM file. 
    _dcm_file_path: str
        Full path where the DICOM file is stored.
    """
    
    def __init__(self, gcs_url) -> None:
        """
        Constructor of WSILevel.    

        Parameters
        ----------
        gcs_url: str
            GCS URL of the DICOM file. 
        """ 
        self._dcm_image, self._dcm_file_path = WSILevel._open_dcm_image(gcs_url)

    def __del__(self):
        """
        Destructor of WSILevel. 
        """ 
        self._dcm_image.close()
        os.remove(self._dcm_file_path)

    @staticmethod
    def _download_dcm_file(dcm_file_path: str, gcs_url: str) -> None: 
        """
        Downloads a DICOM file from the IDC in a python subprocess using gsutil.    

        Parameters
        ----------
        dcm_file_path: str
            Full path where the DICOM file is to be stored. 
        gcs_url: str 
            GCS URL of the DICOM file to be downloaded.
        """ 
        subprocess.run(['gsutil', 'cp', gcs_url, os.path.dirname(dcm_file_path)], 
                       check=True)
    
    @staticmethod
    def _open_dcm_image(gcs_url: str) -> Tuple['WsiDicom', str]:
        """
        Opens a DICOM file using the WsiDicom library.     

        Parameters
        ---------- 
        gcs_url: str 
            GCS URL of the DICOM file to be downloaded.

        Returns
        -------
        Tuple
            Tuple consisting of the WsiDicom object and the full path where the DICOM file is stored.
        """ 
        dcm_file_path = os.path.join('/tmp', os.path.basename(gcs_url))
        if not os.path.isfile(dcm_file_path):
            WSILevel._download_dcm_file(dcm_file_path, gcs_url)
        return (WsiDicom.open(dcm_file_path), dcm_file_path)

    def get_region(self, pos: Tuple[int, int], size: Tuple[int, int]) -> Image.Image:
        """
        Gets a certain region of a WSI defined by pixels.      

        Parameters
        ---------- 
        pos: tuple 
            Upper left corner of the region in pixels. 
        size: tuple
            Size of the requested region in pixels. 

        Returns
        -------
        Image.Image
            Requested region of the WSI. 
        """ 
        return self._dcm_image.read_region(level=0, location=pos, size=size)


class WSI:
    """
    A class representing a whole-slide image.

    Attributes
    ----------
    _slide_metadata: pd.DataFrame
         One-row dataframe containing metadata of the slide of interest.
    _levels: list
        List holding metadata about the different levels available for the WSI of interest.
    """
    
    def __init__(self, slide_metadata: pd.DataFrame) -> None:
        """
        Constructor of WSI.    

        Parameters
        ----------
        slide_metadata: pd.DataFrame
            One-row dataframe containing metadata of the slide of interest.
        """  
        self._slide_metadata = slide_metadata
        self._levels = [None] * len(self._slide_metadata['levels'])

    def __del__(self):
        """
        Destructor of WSI.
        """ 
        for level in self._levels:
            if level:
                del level

    @staticmethod
    def _get_idealized_pixel_spacing(actual_pixel_spacing: float) -> float:
        """
        Determines the idealized pixel spacing (one of 0.008, 0.004, 0.002, 0.001, 0.0005, 0.00025 mm/px), 
        which is closest to the real pixel spacing.

        Parameters
        ---------- 
        actual_pixel_spacing: float
            Actual pixel spacing

        Returns
        -------
        float
            Idealized pixel spacing. 
        """ 
        # Pixel spacings corresponding to 1.25x, 2.5x, 10x, 5x, 20x, 40x objective magnificaton
        pixel_spacings = [0.008, 0.004, 0.002, 0.001, 0.0005, 0.00025]
        for pixel_spacing in pixel_spacings:
            # Allow some tolerance
            if actual_pixel_spacing >= pixel_spacing * 0.7 and\
               actual_pixel_spacing <= pixel_spacing * 1.3: 
                return pixel_spacing
        raise RuntimeError('No idealized pixel_spacing found for {}'.format(actual_pixel_spacing))

    def _get_scale_factor(self, level_index: int, required_pixel_spacing: float) -> float:
        """
        Determines scale factor between the pixel spacing at a given level and the required pixel spacing.

        Parameters
        ---------- 
        level_index: int
            Zero-based index of level in the WSI pyramid.
        required_pixel_spacing: float
            Required pixel spacing in mm/px.

        Returns
        -------
        float
            Scale factor. 
        """ 
        level_pixel_spacing = self._slide_metadata['levels'][level_index]['pixel_spacing']
        return required_pixel_spacing / WSI._get_idealized_pixel_spacing(level_pixel_spacing)
    
    def get_size_at_pixel_spacing(self, pixel_spacing: float) -> Tuple[int, int]: 
        """
        Get the image size of the WSI when scaled to a have a certain pixel spacing. 

        Parameters
        ---------- 
        pixel_spacing: float
            Pixel spacing of interest.

        Returns
        -------
        tuple
            (width, hight) of image in pixels. 
        """ 
        scale_factor = self._get_scale_factor(0, pixel_spacing)
        return (int(self._slide_metadata['width'] / scale_factor),
                int(self._slide_metadata['height'] / scale_factor))

    def _determine_level(self, pixel_spacing: float) -> int:
        """
        Find level in the WSI pyramid which is closest to a certain pixel spacing. 

        Parameters
        ---------- 
        pixel_spacing: float
            Pixel spacing of interest.

        Returns
        -------
        int
            Zero-based index of level in the WSI pyramid.
        """ 
        levels_data = self._slide_metadata['levels']
        for level_index in reversed(range(len(levels_data))):
            try:
                level_pixel_spacing = WSI._get_idealized_pixel_spacing(levels_data[level_index]['pixel_spacing'])
                if level_pixel_spacing <= pixel_spacing:
                    return level_index
            except RuntimeError:
                pass

        raise RuntimeError('no level for pixel spacing {} in slide {}'.format(\
            pixel_spacing, self._slide_metadata['digital_slide_id']))

    def _lazy_load_level(self, level_index: int) -> 'WSILevel':
        """
        Loads a certain level of a WSI.    

        Parameters
        ---------- 
        level: int
            Zero-based index of level in the WSI pyramid.

        Returns
        -------
        WSILevel
            WSILevel object of the level requested.
        """ 
        if not self._levels[level_index]:
            level_data = self._slide_metadata['levels'][level_index]
            self._levels[level_index] = WSILevel(level_data['gcs_url'])
        return self._levels[level_index]

    def _scale_to_level(self, pos, size, level_index, pixel_spacing) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Scales a given image region at a given level to another level which is defined by its pixel spacing.    

        Parameters
        ---------- 
        pos: tuple 
            Upper left corner of the region in pixels. 
        size: tuple
            Size of the region in pixels.
        level_index: int
            Zero-based index of level in the WSI pyramid to which pos and size relate.
        pixel_spacing:
            Pixel spacing to which the region should be scaled.

        Returns
        -------
        tuple
            Tuple of the scaled pos and scaled size. 
        """
        scale_factor = self._get_scale_factor(level_index, pixel_spacing)
        
        scaled_pos_1 = [pos[0] * scale_factor, pos[1] * scale_factor]
        scaled_pos_2 = [scaled_pos_1[0] + size[0] * scale_factor, 
                        scaled_pos_1[1] + size[1] * scale_factor]
        
        # Clip against level size
        level_data = self._slide_metadata['levels'][level_index]
        level_size = (level_data['width'], level_data['height'])
        scaled_pos_1[0] = max(scaled_pos_1[0], 0)
        scaled_pos_2[0] = min(scaled_pos_2[0], level_size[0])
        scaled_pos_1[1] = max(scaled_pos_1[1], 0)
        scaled_pos_2[1] = min(scaled_pos_2[1], level_size[1])

        scaled_size = (scaled_pos_2[0] - scaled_pos_1[0], scaled_pos_2[1] - scaled_pos_1[1])
        
        scaled_pos_1, scaled_size = [int(x) for x in scaled_pos_1], [int(x) for x in scaled_size]
        return (tuple(scaled_pos_1), tuple(scaled_size))

    def get_region(self, pos: Tuple[int, int], size: Tuple[int, int], pixel_spacing: float) -> Image.Image:
        """
        Gets a certain region of a WSI defined by pixels at a certain pixel spacing.      

        Parameters
        ---------- 
        pos: tuple 
            Upper left corner of the region in pixels. 
        size: tuple
            Size of the requested region in pixels. 
        pixel_spacing: float
            Pixel spacing to which pos and size relate.

        Returns
        -------
        Image.Image
            Requested region of the WSI. 
        """ 
        level_index = self._determine_level(pixel_spacing)
        scaled_pos, scaled_size = self. _scale_to_level(pos, size, level_index, pixel_spacing)
        scaled_region = self._lazy_load_level(level_index).get_region(scaled_pos, scaled_size)
        return scaled_region.resize(size)

    def get_tile(self, tile_pos: Tuple[int, int], tile_size: int, pixel_spacing: float) -> Image.Image:
        """
        Returns a requested tile of the WSI.      

        Parameters
        ---------- 
        tile_pos: tuple 
            Position of the tile in the WSI with (0,0) refering to the tile in the upper left corner. 
        tile_size: tuple
            Size of the tile in pixels. 
        pixel_spacing: float
            Pixel spacing of the WSI at which the tile should be extracted.

        Returns
        -------
        Image.Image
            Requested tile. 
        """ 
        pos = tile_pos[0] * tile_size, tile_pos[1] * tile_size
        size = (tile_size, tile_size)
        return self.get_region(pos, size, pixel_spacing)

    def get_thumbnail(self, thumbnail_width: int=300) -> Image.Image:
        """
        Gets a thumbnail image of the WSI. 

        Parameters
        ---------- 
        thumbnail_width: int 
            Required width of the thumbnail in pixels. Height will be adapted according to the image relations.

        Returns
        -------
        Image.Image
            Thumbail image.  
        """ 
        image_width = self._slide_metadata['width']
        factor = thumbnail_width / image_width
        thumbnail_height = int(self._slide_metadata['height'] * factor)
        pixel_spacing = self._slide_metadata['levels'][0]['pixel_spacing'] / factor
        thumbnail_size = (thumbnail_width, thumbnail_height)
        return self.get_region((0,0), thumbnail_size, pixel_spacing)


class WSIOpener:
    """
    Class to manage a set of WSI.

    Attributes
    ----------
    _slides_metadata: pd.DataFrame
        Slides metadata table with one row per slide. 
    """

    def __init__(self, slides_metadata: pd.DataFrame):
        """
        Constructor of WSIOpener.    

        Parameters
        ----------
        slides_metadata: pd.DataFrame
             Slides metadata table with one row per slide. 
        """
        self._slides_metadata = slides_metadata

    def open_image(self, image_id: str) -> WSI:
        """
        Creates a WSI object given an image ID of interest.    

        Parameters
        ----------
        image_id: str
            Image ID of interest. 
        
        Returns 
        -------
        WSI
            WSI object for provided image ID. 
        """
        return WSI(self._slides_metadata.loc[image_id])

    def get_image_ids(self) -> List[str]:
        """
        Returns all image IDs organized in the WSIOpener instance.    
        
        Returns 
        -------
        list
            List of image IDs. 
        """
        return self._slides_metadata['digital_slide_id'].tolist()
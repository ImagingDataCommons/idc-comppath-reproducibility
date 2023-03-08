import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
from .tile_list import TileList


def split_tile_list(tile_list: TileList, slides_metadata: pd.DataFrame, proportions: List[float], random_state: int) -> Tuple[TileList]:
    """
    Subdivides a given TileList object into a TileList for training, validation and testing, respectively.   

    Parameters
    ----------
    tile_list: TileList
        TileList object containing all relevant tiles.
    slides_metadata: pd.DataFrame
        Slides metadata table with one row per slide.
    proportions: list
        List of floats that sum up to 1.0 and define the requested proportions of tiles in the 
        training, validation and test set, respectively.
    random_state: int 
        Integer used to seed the pd.sample function.  

    Returns 
    -------
    tuple 
        Tuple containing TileList instances for training, validation and testing. 
    """
    # Shuffle slides metadata according to random_state
    slides_metadata = slides_metadata.sample(frac=1.0, random_state=random_state)

    # Determine image_ids per patient
    patient_id_to_image_ids_map = defaultdict(list)
    for _, slide_metadata in slides_metadata.iterrows(): 
        patient_id_to_image_ids_map[slide_metadata['patient_id']].append(slide_metadata['digital_slide_id']) 
    
    # Determine patient_ids per cancer subtype
    cancer_subtype_to_patient_ids_map = defaultdict(list) 
    for patient_id, image_ids in patient_id_to_image_ids_map.items():
        for img_id in image_ids: 
            if slides_metadata.loc[img_id]['reference_class_label'] == 'normal':
                contains_normal = True
                break
            contains_normal = False
        cancer_subtype = slides_metadata.loc[img_id]['cancer_subtype']
        cancer_subtype_to_patient_ids_map[(cancer_subtype, contains_normal)].append(patient_id)
    
    # Assign patients to subsets stratified by cancer subtype 
    subset_to_patient_ids_map = _assign_patients_to_subsets(cancer_subtype_to_patient_ids_map, proportions)
    
    # Create corresponding tile lists
    tile_lists = []
    for i in range(len(proportions)):
        img_ids_subset = [img_id 
                          for patient_id in subset_to_patient_ids_map[i] 
                          for img_id in patient_id_to_image_ids_map[patient_id]]
        tile_lists.append(tile_list.subset(img_ids_subset))
    return tuple(tile_lists)


def _assign_patients_to_subsets(cancer_subtype_to_patient_ids_map: Dict[Tuple[str,bool],List[str]], proportions: List[float]) -> List[List[str]]:
    """
    Assigns patients to training, validation or test set.

    Parameters
    ----------
    cancer_subtype_to_patient_ids_map: dict
        Dictionary mapping cancer subtype to a list of patients diagnosed with that cancer subtype.
    proportions: list
        List of floats that sum up to 1.0 and define the requested proportions of tiles in the 
        training, validation and test set, respectively.  

    Returns 
    -------
    list 
        List with three sublists, each containing all patients assigned to the training, validation and test set, respectively. 
    """
    # Assign patients to subsets stratified by cancer subtype 
    subset_to_patient_ids_list = [[] for i in range(len(proportions))]
    for patient_ids in cancer_subtype_to_patient_ids_map.values():
        subset_idx = [0] + [int(sum(proportions[:i+1]) / sum(proportions) * len(patient_ids))\
                            for i in range(len(proportions))]
        for i in range(len(proportions)):
            subset_to_patient_ids_list[i].extend(patient_ids[subset_idx[i]:subset_idx[i+1]])
    return subset_to_patient_ids_list
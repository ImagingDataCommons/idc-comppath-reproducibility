import random
import pandas as pd
from typing import Any, Dict


def _get_reference_class_label(slide_metadata: pd.DataFrame) -> str:
    """
    Gets the reference class label of a certain slide.

    Parameters
    ----------
    slide_metadata: pd.DataFrame
        One-row dataframe containing metadata of the slide of interest

    Returns
    -------
    str
        String describing the tissue type ('normal', 'luad', 'lssc')
    """ 
    tissue_type = slide_metadata['tissue_type']
    if tissue_type == 'normal':
        return tissue_type
    else: 
        return slide_metadata['cancer_subtype']


def create_slides_metadata(bq_results_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]: 
    """
    Builds a dataframe comprising all slides' metadata. 

    Parameters
    ----------
    bq_results_df: pd.DataFrame
        Dataframe obtained from BigQuery. Contains one DICOM file (one level of a slide) per row. 

    Returns
    -------
    pd.DataFrame
        Slides metadata table with one row per slide. 
    """
    slides_metadata = dict()

    for index, row in bq_results_df.iterrows():
        slide_metadata = row.to_dict()
        image_id = slide_metadata['digital_slide_id']
        
        # Move level specific values through "pop()"
        level_data = {
            'width': slide_metadata.pop('width', None),
            'height': slide_metadata.pop('height', None),
            'pixel_spacing': slide_metadata.pop('pixel_spacing', None),
            'compression': slide_metadata.pop('compression', None),
            'crdc_instance_uuid': slide_metadata.pop('crdc_instance_uuid', None),
            'gcs_url': slide_metadata.pop('gcs_url', None)
        }

        if not image_id in slides_metadata:
            slides_metadata[image_id] = slide_metadata
            slides_metadata[image_id]['reference_class_label'] = _get_reference_class_label(slide_metadata)
            slides_metadata[image_id]['levels'] = []

        slides_metadata[image_id]['levels'].append(level_data)

    for slide_metadata in slides_metadata.values():
        slide_metadata['levels'].sort(key=lambda x: x['pixel_spacing'])
    
        if len(slide_metadata['levels']) > 0:
            base_level = slide_metadata['levels'][0]
            slide_metadata['width'] = base_level['width'] 
            slide_metadata['height'] = base_level['height'] 

    return pd.DataFrame.from_records(list(slides_metadata.values()),
                                     index=list(slides_metadata.keys()))


def get_stratified_subsample(slides_metadata: pd.DataFrame, num_slides: int, random_state: int) -> pd.DataFrame:
    """
    Gets a subsample from slides_metadata with the same amount of samples for each class.   

    Parameters
    ----------
    slides_metadata: pd.DataFrame
        Slides metadata table with one row per slide. 
    num_slides: int 
        Number of slides which should be returned in total. 
    random_state: int 
        Integer used to seed the pd.sample function. 

    Returns
    -------
    pd.DataFrame
        Subsample of slides metadata table.  
    """
    assert num_slides % NUM_CLASSES == 0
    return pd.concat(\
        [slides_metadata.loc[slides_metadata['reference_class_label'] == cl]\
         .sample(num_slides // NUM_CLASSES, random_state=random_state)
            for cl in CLASS_LABEL_TO_INDEX_MAP.keys()])
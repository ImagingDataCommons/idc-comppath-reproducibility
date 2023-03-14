def perform_tile_based_roc_analysis(predictions: Predictions) -> ROCAnalysis:
    """
    Function that initiates running a tile-based ROC analysis.     
    
    Parameters
    ---------- 
    predictions: Predictions
        Predictions object to work with. 
    
    Returns
    -------
    tuple
       ROCAnalsis object. 
    """ 
    return ROCAnalysis('tile-based', 
                       predictions._predictions['reference_class_index'], 
                       predictions._predictions['predicted_class_probabilities'])

def perform_slide_based_roc_analysis(predictions: Predictions) -> ROCAnalysis:
    """
    Function that initiates running a tile-based ROC analysis.     
    
    Parameters
    ---------- 
    predictions: Predictions
        Predictions object to work with. 
    
    Returns
    -------
    tuple
       ROCAnalsis object. 
    """ 
    # Average predictions of tiles to obtain one prediction per image
    all_image_ids = predictions.get_all_image_ids()
    reference_classes = np.empty(len(all_image_ids))
    class_probabilities = np.empty((len(all_image_ids), NUM_CLASSES))

    for i, image_id in enumerate(all_image_ids):
        image_predictions = predictions.get_results_of_image(image_id)
        reference_classes[i] = image_predictions['reference_class_index'].to_numpy()[0]
        image_class_probabilities = np.stack(image_predictions['predicted_class_probabilities'])
        class_probabilities[i] = np.average(image_class_probabilities, axis=0)
    
    return ROCAnalysis('slide-based', 
                       reference_classes, 
                       class_probabilities)
    
def summarize_roc_values(roc_analysis1: ROCAnalysis, roc_analysis2: ROCAnalysis) -> None:
    """
    Summarizes results from two separate ROCAnalysis objects.     
    
    Parameters
    ---------- 
    roc_analysis1: ROCAnalysis
    roc_analysis2: ROCAnalysis
    """ 
    class_to_str_map = {i: l for l, i in CLASS_LABEL_TO_INDEX_MAP.items()}
    results_dict = {(roc_analysis1.name, 'auc'): roc_analysis1.auc, 
                    (roc_analysis1.name, 'confidence'): roc_analysis1.ci,
                    (roc_analysis2.name, 'auc'): roc_analysis2.auc, 
                    (roc_analysis2.name, 'confidence'): roc_analysis2.ci}
    results = pd.DataFrame(results_dict)
    results.rename(index=class_to_str_map, inplace=True)
    return results.round(decimals=3)
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.preprocessing import label_binarize
from typing import List, Tuple
from .predictions import Predictions
from .global_variables import CLASS_LABEL_TO_INDEX_MAP, NUM_CLASSES


class ROCAnalysis():
    """
    Class for the ROC analysis. 

    Attributes
    ----------
    name: str
        Name of the ROC analysis.
    auc: dict
        Dictionary containing Area Under the Curve (AUC) values.
    ci: dict
        Dictionary containing confidence intervals. 
    fpr: dict
        Dictionary containing False Positive Rate values,
    tpr: dict
        Dictionary containing True Positive Rate values.
    """
    
    def __init__(self, name: str, reference_classes: np.ndarray, class_probabilities: np.ndarray) -> None:
        """
        Constructor of ROCAnalysis.  

        Parameters
        ----------
        name: str
            Name of the ROC analysis.  
        reference_classes: np.ndarray
            Array containing the reference class values of shape (1, n)
        class_probabilites: np.ndarray
            Array containing the class probability values as predicted by the network. Shape (num_classes, n).   
        """ 
        self.name = name
        self.auc, self.ci, self.fpr, self.tpr = {}, {}, {}, {}
        self._generate_multiclass_roc_curves(reference_classes, class_probabilities)  

    def _generate_multiclass_roc_curves(self, reference_classes: np.ndarray, class_probabilities: np.ndarray) -> None: 
        """
        Function computing multclass ROC curves, i.e. a 1 vs. the rest curve for each class.   

        Parameters
        ---------- 
        reference_classes: np.ndarray
            Array containing the reference class values of shape (1, n)
        class_probabilites: np.ndarray
            Array containing the class probability values as predicted by the network. Shape (num_classes, n).   
        """     
        # Binarize the reference values (one-hot-encoding)
        reference = label_binarize(reference_classes, classes=[i for i in range(NUM_CLASSES)])

        for i in range(NUM_CLASSES):
            prediction = np.asarray([cp[i] for cp in class_probabilities])
            self.fpr[i], self.tpr[i], self.auc[i] = self._generate_roc_curve(reference[:,i], prediction)
            self.ci[i] = self._get_confidence_interval_by_bootstrapping(reference[:,i], prediction)

    def _get_confidence_interval_by_bootstrapping(self, reference: np.ndarray, prediction: np.ndarray, 
                                                  num_bootstraps: int = 1000) -> List[float]: 
        """
        Function computing confidence intervals by bootstrapping.    

        Parameters
        ---------- 
        reference: np.ndarray
            Array containing the one-hot encoded reference class values. Shape (1, n)
        prediction: np.ndarray
            Array containing the predicted values. Shape (1, n).  
        num_bootstraps: int
            Number of bootstrapping iterations. Defaults to 1000.

        Returns
        -------
        list: 
            List containig two values: the lower and the upper bound of the confidence interval.
        """        
        bootstrap_scores = np.empty(num_bootstraps)
        i = 0
        while i < num_bootstraps:
            bootstrap_indices = np.random.randint(0, len(reference), size=len(reference))
            reference_sample = reference[bootstrap_indices]
            prediction_sample = prediction[bootstrap_indices]
            # We need at least one positive and one negative sample
            if len(np.unique(reference_sample)) < 2:
                continue
            else: 
                bootstrap_scores[i] = skm.roc_auc_score(reference_sample, prediction_sample)
                i += 1

        bootstrap_scores.sort()
        ci_lower = bootstrap_scores[int(0.025 * len(bootstrap_scores))]
        ci_upper = bootstrap_scores[int(0.975 * len(bootstrap_scores))]
        return [ci_lower, ci_upper] 

    def _generate_roc_curve(self, reference: np.ndarray, prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Function computing confidence intervals by bootstrapping.    

        Parameters
        ---------- 
        reference: np.ndarray
            Array containing the one-hot encoded reference class values. Shape (1, n)
        prediction: np.ndarray
            Array containing the predicted values. Shape (1, n).    

        Returns
        -------
        tuple: 
            A tuple containing FPR values, TPR values and the AUC value.
        """ 
        fpr, tpr, _ = skm.roc_curve(
            y_true = reference,
            y_score = prediction
        )
        auc = skm.auc(fpr, tpr)
        return fpr, tpr, auc

    def plot(self, axis: plt.Axes) -> None:
        """
        Plots results of the ROC analysis.    

        Parameters
        ---------- 
        axis: plt.Axes
            Subplot to be used for plotting the analysis results.
        """ 
        colors = ['g', 'b', 'orange']

        self._plot_bisector(axis)
        class_to_str_map = {i: l for l, i in CLASS_LABEL_TO_INDEX_MAP.items()}
        for idx, key in enumerate(class_to_str_map): 
            label='{} (AUC = {:.3f})'.format(class_to_str_map[key], self.auc[key])
            axis.plot(
                self.fpr[key],
                self.tpr[key],
                color=colors[idx],
                linewidth=2,
                label=label
            )
        self._format_plot(axis, self.name)

    def _plot_bisector(self, axis: plt.Axes) -> None:
        """
        Plots the bisector along with the ROC curve.    

        Parameters
        ---------- 
        axis: plt.Axes
            Subplot to be used for plotting the analysis results.
        """ 
        axis.plot(
            [0, 1],
            [0, 1],
            color='black',
            linewidth=1,
            linestyle='--'
        )

    def _format_plot(self, axis: plt.Axes, title: str) -> None:
        """
        Formats the final plot.    

        Parameters
        ---------- 
        axis: plt.Axes
            Subplot to be used for plotting the analysis results.
        title: str
            Title to be used for the plot. 
        """ 
        axis.set_title(title)
        axis.set_xlim([0.0, 1.0])
        axis.set_ylim([0.0, 1.0])
        axis.set_xlabel('False positive rate')
        axis.set_ylabel('True positive rate')
        axis.legend(loc='lower right')


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

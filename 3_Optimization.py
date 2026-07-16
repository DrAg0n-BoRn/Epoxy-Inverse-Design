from ml_tools.ML_optimization import DragonOptimizer
from ml_tools.ML_configuration import DragonOptimizerConfig
from ml_tools.ML_models import DragonNodeModel
from ml_tools.ML_inference import DragonInferenceHandler
from ml_tools.ML_utilities import DragonArtifactFinder
from ml_tools.schema import FeatureSchema

import torch

from paths import PM
from helpers.constants import TARGET


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    # Set up the optimizer configuration
    config = DragonOptimizerConfig(target_name=TARGET,
                                   task="max",
                                   continuous_bounds_map=PM.artifacts,
                                   save_directory=PM.optimization,
                                   save_format="csv",
                                   algorithm="Genetic",
                                   population_size=25,
                                   generations=10,
                                   repetitions=500)
    
    # Load train artifacts
    artifact_finder = DragonArtifactFinder(directory=PM.regression,
                                           load_scaler=True,
                                           load_schema=True,
                                           strict=True)
    
    # Load the model architecture
    model = DragonNodeModel.load_architecture(artifact_finder.model_architecture_path) # type: ignore
    
    # Load the model weights and set up the inference handler
    inference_handler = DragonInferenceHandler(model=model,
                                               state_dict=artifact_finder.weights_path, # type: ignore
                                               device=DEVICE,
                                               scaler=artifact_finder.scaler_path)
    
    # Set up the optimizer
    optimizer = DragonOptimizer(inference_handler=inference_handler,
                                schema=artifact_finder.feature_schema, # type: ignore
                                config=config)
    
    optimizer.run()


if __name__ == "__main__":
    main()

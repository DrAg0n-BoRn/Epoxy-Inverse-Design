from ml_tools.path_manager import DragonPathManager


# 1. Initialize the PathManager using this file as the anchor, adding base directories.
PM = DragonPathManager(
    anchor_file=__file__,
    base_directories=["helpers", "start_data", "results", "backups"]
)

# 2. Define directories and files.
### Base files
PM.imputed_file = PM.start_data / "imputed_data.csv"
PM.diffusion_generated = PM.start_data / "Diffusion Generated"

### Datasets
PM.artifacts = PM.results / "Artifacts"

### Regression
PM.regression = PM.results / "Regression"

### Optimization
PM.optimization = PM.results / "Optimization"

#### Comparison data
PM.comparison = PM.results / "Generated Comparison"

# 3. Make directories and check status
PM.make_dirs()

if __name__ == "__main__":
    PM.status()

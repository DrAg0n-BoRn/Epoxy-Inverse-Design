from ml_tools.schema import FeatureSchema
from ml_tools.optimization_tools import make_continuous_bounds_template

from paths import PM


if __name__ == "__main__":
    # Load the schema
    schema = FeatureSchema.from_json(PM.start_data)

    # Make the bounds template
    make_continuous_bounds_template(
        directory=PM.artifacts,
        feature_schema=schema,
    )

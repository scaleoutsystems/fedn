from fedn.network.api.v1.analytic_routes import bp as analytic_bp
from fedn.network.api.v1.client_routes import bp as client_bp
from fedn.network.api.v1.combiner_routes import bp as combiner_bp
from fedn.network.api.v1.helper_routes import bp as helper_bp
from fedn.network.api.v1.metric_routes import bp as metric_bp
from fedn.network.api.v1.model_routes import bp as model_bp
from fedn.network.api.v1.package_routes import bp as package_bp
from fedn.network.api.v1.prediction_routes import bp as prediction_bp
from fedn.network.api.v1.round_routes import bp as round_bp
from fedn.network.api.v1.run_routes import bp as training_run_bp
from fedn.network.api.v1.session_routes import bp as session_bp
from fedn.network.api.v1.status_routes import bp as status_bp
from fedn.network.api.v1.telemetry_routes import bp as telemetry_bp
from fedn.network.api.v1.validation_routes import bp as validation_bp

_routes = [
    client_bp,
    combiner_bp,
    model_bp,
    package_bp,
    round_bp,
    session_bp,
    status_bp,
    validation_bp,
    prediction_bp,
    helper_bp,
    analytic_bp,
    metric_bp,
    training_run_bp,
    telemetry_bp,
]

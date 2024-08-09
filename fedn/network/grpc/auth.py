import grpc
import jwt

from fedn.common.config import FEDN_AUTH_SCHEME, FEDN_JWT_ALGORITHM, SECRET_KEY
from fedn.common.log_config import logger
from fedn.network.api.auth import check_custom_claims

ENDPOINT_ROLES_MAPPING = {
    "/fedn.Combiner/TaskStream": ["client"],
    "/fedn.Combiner/SendModelUpdate": ["client"],
    "/fedn.Combiner/SendModelValidation": ["client"],
    "/fedn.Connector/SendHeartbeat": ["client"],
    "/fedn.Connector/SendStatus": ["client"],
    "/fedn.ModelService/Download": ["client"],
    "/fedn.ModelService/Upload": ["client"],
    "/fedn.Control/Start": ["controller"],
    "/fedn.Control/Stop": ["controller"],
    "/fedn.Control/FlushAggregationQueue": ["controller"],
    "/fedn.Control/SetAggregator": ["controller"],
}

ENDPOINT_WHITELIST = [
    "/fedn.Connector/AcceptingClients",
    "/fedn.Connector/ListActiveClients",
    "/fedn.Control/Start",
    "/fedn.Control/Stop",
    "/fedn.Control/FlushAggregationQueue",
    "/fedn.Control/SetAggregator",
]

USER_AGENT_WHITELIST = ["grpc_health_probe"]


def check_role_claims(payload, endpoint):
    user_role = payload.get("role", "")

    # Perform endpoint-specific RBAC check
    allowed_roles = ENDPOINT_ROLES_MAPPING.get(endpoint)
    if allowed_roles and user_role not in allowed_roles:
        return False
    return True


def _unary_unary_rpc_terminator(code, details):
    def terminate(ignored_request, context):
        context.abort(code, details)

    return grpc.unary_unary_rpc_method_handler(terminate)


class JWTInterceptor(grpc.ServerInterceptor):
    def __init__(self):
        pass

    def intercept_service(self, continuation, handler_call_details):
        # Pass if no secret key is set
        if not SECRET_KEY:
            return continuation(handler_call_details)
        metadata = dict(handler_call_details.invocation_metadata)
        # Pass whitelisted methods
        if handler_call_details.method in ENDPOINT_WHITELIST:
            return continuation(handler_call_details)
        # Pass if the request comes from whitelisted user agents
        user_agent = metadata.get("user-agent").split(" ")[0]
        if user_agent in USER_AGENT_WHITELIST:
            return continuation(handler_call_details)

        token = metadata.get("authorization")
        if token is None:
            return _unary_unary_rpc_terminator(grpc.StatusCode.UNAUTHENTICATED, "Token is missing")

        if not token.startswith(FEDN_AUTH_SCHEME):
            return _unary_unary_rpc_terminator(grpc.StatusCode.UNAUTHENTICATED, f"Invalid token scheme, expected {FEDN_AUTH_SCHEME}")

        token = token.split(" ")[1]

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[FEDN_JWT_ALGORITHM])

            if not check_role_claims(payload, handler_call_details.method):
                return _unary_unary_rpc_terminator(grpc.StatusCode.PERMISSION_DENIED, "Insufficient permissions")

            if not check_custom_claims(payload):
                return _unary_unary_rpc_terminator(grpc.StatusCode.PERMISSION_DENIED, "Insufficient permissions")

            return continuation(handler_call_details)
        except jwt.InvalidTokenError:
            return _unary_unary_rpc_terminator(grpc.StatusCode.UNAUTHENTICATED, "Invalid token")
        except jwt.ExpiredSignatureError:
            return _unary_unary_rpc_terminator(grpc.StatusCode.UNAUTHENTICATED, "Token expired")
        except Exception as e:
            logger.error(str(e))
            return _unary_unary_rpc_terminator(grpc.StatusCode.UNKNOWN, str(e))

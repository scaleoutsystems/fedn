from functools import wraps

import jwt
from flask import jsonify, request

from fedn.common.config import (
    FEDN_AUTH_SCHEME,
    FEDN_AUTH_WHITELIST_URL_PREFIX,
    FEDN_JWT_ALGORITHM,
    FEDN_JWT_CUSTOM_CLAIM_KEY,
    FEDN_JWT_CUSTOM_CLAIM_VALUE,
    SECRET_KEY,
)


def check_role_claims(payload, role):
    if "role" not in payload:
        return False
    if payload["role"] != role:
        return False

    return True


def check_custom_claims(payload):
    if FEDN_JWT_CUSTOM_CLAIM_KEY and FEDN_JWT_CUSTOM_CLAIM_VALUE:
        if payload[FEDN_JWT_CUSTOM_CLAIM_KEY] != FEDN_JWT_CUSTOM_CLAIM_VALUE:
            return False
    return True


def if_whitelisted_url_prefix(path):
    if FEDN_AUTH_WHITELIST_URL_PREFIX and path.startswith(FEDN_AUTH_WHITELIST_URL_PREFIX):
        return True
    else:
        return False


def jwt_auth_required(role=None):
    def actual_decorator(func):
        if not SECRET_KEY:
            return func

        @wraps(func)
        def decorated(*args, **kwargs):
            if if_whitelisted_url_prefix(request.path):
                return func(*args, **kwargs)
            token = request.headers.get("Authorization")
            if not token:
                return jsonify({"message": "Missing token"}), 401
            # Get token from the header Bearer
            if token.startswith(FEDN_AUTH_SCHEME):
                token = token.split(" ")[1]
            else:
                return jsonify({"message": f"Invalid token scheme, expected {FEDN_AUTH_SCHEME}"}), 401
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[FEDN_JWT_ALGORITHM])
                if not check_role_claims(payload, role):
                    return jsonify({"message": "Invalid token"}), 401
                if not check_custom_claims(payload):
                    return jsonify({"message": "Invalid token"}), 401

            except jwt.ExpiredSignatureError:
                return jsonify({"message": "Token expired"}), 401

            except jwt.InvalidTokenError:
                return jsonify({"message": "Invalid token"}), 401

            return func(*args, **kwargs)

        return decorated

    return actual_decorator

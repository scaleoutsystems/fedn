import os
from functools import wraps

import jwt
from flask import jsonify, request

# Define your secret key for JWT
SECRET_KEY = os.environ.get('FEDN_JWT_SECRET_KEY', False)
FEDN_JWT_CUSTOM_CLAIM_KEY = os.environ.get('FEDN_JWT_CUSTOM_CLAIM_KEY', False)
FEDN_JWT_CUSTOM_CLAIM_VALUE = os.environ.get('FEDN_JWT_CUSTOM_CLAIM_VALUE', False)
FEDN_AUTH_SCHEME = os.environ.get('FEDN_AUTH_SCHEME', 'Bearer')


def check_role_claims(payload, role):
    if FEDN_JWT_CUSTOM_CLAIM_KEY and FEDN_JWT_CUSTOM_CLAIM_VALUE:
        if payload[FEDN_JWT_CUSTOM_CLAIM_KEY] != FEDN_JWT_CUSTOM_CLAIM_VALUE:
            return False
    if 'role' not in payload:
        return False
    if payload['role'] != role:
        return False

    return True


def check_custom_claims(payload):
    if FEDN_JWT_CUSTOM_CLAIM_KEY and FEDN_JWT_CUSTOM_CLAIM_VALUE:
        if payload[FEDN_JWT_CUSTOM_CLAIM_KEY] != FEDN_JWT_CUSTOM_CLAIM_VALUE:
            return False
    return True


def jwt_auth_required(role=None):
    def actual_decorator(func):
        if not SECRET_KEY:
            return func

        @wraps(func)
        def decorated(*args, **kwargs):
            token = request.headers.get('Authorization')
            # Get token from the header Bearer
            if token and token.startswith(FEDN_AUTH_SCHEME):
                token = token.split(' ')[1]

            if not token:
                return jsonify({'message': 'Missing token'}), 401

            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
                if not check_role_claims(payload, role):
                    return jsonify({'message': 'Invalid token'}), 401
                if not check_custom_claims(payload):
                    return jsonify({'message': 'Invalid token'}), 401

            except jwt.ExpiredSignatureError:
                return jsonify({'message': 'Token expired'}), 401

            except jwt.InvalidTokenError:
                return jsonify({'message': 'Invalid token'}), 401

            return func(*args, **kwargs)

        return decorated
    return actual_decorator

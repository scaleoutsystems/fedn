import threading

from flask import Blueprint, jsonify, request

from fedn.common.log_config import logger
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.v1.shared import api_version, get_post_data_to_kwargs, get_typed_list_headers
from fedn.network.combiner.interfaces import CombinerUnavailableError
from fedn.network.controller.control import Control
from fedn.network.state import ReducerState
from fedn.network.storage.statestore.stores.dto.session import SessionConfigDTO, SessionDTO
from fedn.network.storage.statestore.stores.shared import EntityNotFound, MissingFieldError, ValidationError

bp = Blueprint("session", __name__, url_prefix=f"/api/{api_version}/sessions")


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
def get_sessions():
    """Get sessions
    Retrieves a list of sessions based on the provided parameters.
    By specifying a parameter in the url, you can filter the sessions based on that parameter,
    and the response will contain only the sessions that match the filter.
    ---
    tags:
        - Sessions
    parameters:
      - name: status
        in: query
        required: false
        type: string
        description: Status of the session
      - name: session_id
        in: query
        required: false
        type: string
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of sessions to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of sessions to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the sessions by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the sessions in ('asc' or 'desc')
    definitions:
      Session:
        type: object
        properties:
          id:
            type: string
          status:
            type: string
          session_id:
            type: string
          session_config:
            type: object
    responses:
      200:
        description: A list of sessions and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Session'
      500:
        description: An error occurred
        schema:
            type: object
            properties:
                message:
                    type: string
    """
    try:
        db = Control.instance().db
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        sessions = db.session_store.list(limit, skip, sort_key, sort_order, **kwargs)

        count = db.session_store.count(**kwargs)
        result = [session.to_dict() for session in sessions]

        response = {"count": count, "result": result}
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/list", methods=["POST"])
@jwt_auth_required(role="admin")
def list_sessions():
    """List sessions
    Retrieves a list of sessions based on the provided parameters.
    Works much like the GET /sessions endpoint, but allows for more complex queries.
    By specifying a parameter in the request body, you can filter the sessions based on that parameter,
    and the response will contain only the sessions that match the filter. If the parameter value contains a comma,
    the filter will be an "in" query, meaning that the sessions will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
      - Sessions
    parameters:
      - name: session
        in: body
        required: false
        schema:
          type: object
          properties:
            status:
              type: string
            session_id:
              type: string
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of sessions to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of sessions to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the sessions by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the sessions in ('asc' or 'desc')
    responses:
      200:
        description: A list of sessions and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Session'
      500:
        description: An error occurred
        schema:
            type: object
            properties:
                message:
                    type: string
    """
    try:
        db = Control.instance().db
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = get_post_data_to_kwargs(request)

        sessions = db.session_store.list(limit, skip, sort_key, sort_order, **kwargs)

        count = db.session_store.count(**kwargs)
        result = [session.to_dict() for session in sessions]

        response = {"count": count, "result": result}

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["GET"])
@jwt_auth_required(role="admin")
def get_sessions_count():
    """Sessions count
    Retrieves the count of sessions based on the provided parameters.
    By specifying a parameter in the url, you can filter the sessions based on that parameter,
    and the response will contain only the count of sessions that match the filter.
    ---
    tags:
        - Sessions
    parameters:
      - name: session_id
        in: query
        required: false
        type: string
      - name: status
        in: query
        required: false
        type: string
    responses:
        200:
            description: The count of sessions
            schema:
                type: integer
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        db = Control.instance().db
        kwargs = request.args.to_dict()
        count = db.session_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["POST"])
@jwt_auth_required(role="admin")
def sessions_count():
    """Sessions count
    Retrieves the count of sessions based on the provided parameters.
    Works much like the GET /sessions/count endpoint, but allows for more complex queries.
    By specifying a parameter in the request body, you can filter the sessions based on that parameter,
    if the parameter value contains a comma, the filter will be an "in" query, meaning that the sessions
    will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
        - Sessions
    parameters:
      - name: session
        in: body
        required: false
        schema:
          type: object
          properties:
            status:
              type: string
            session_id:
              type: string
    responses:
        200:
            description: The count of sessions
            schema:
                type: integer
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        db = Control.instance().db
        kwargs = get_post_data_to_kwargs(request)
        count = db.session_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>", methods=["GET"])
@jwt_auth_required(role="admin")
def get_session(id: str):
    """Get session
    Retrieves a session based on the provided id.
    ---
    tags:
        - Sessions
    parameters:
      - name: id
        in: path
        required: true
        type: string
        description: The id of the session
    responses:
        200:
            description: The session
            schema:
                $ref: '#/definitions/Session'
        404:
            description: The session was not found
            schema:
                type: object
                properties:
                    error:
                        type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        db = Control.instance().db
        result = db.session_store.get(id)
        if result is None:
            return jsonify({"message": f"Entity with id: {id} not found"}), 404
        response = result.to_dict()
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/", methods=["POST"])
@jwt_auth_required(role="admin")
def post():
    """Create session
    Creates a new session based on the provided data.
    ---
    tags:
        - Sessions
    parameters:
      - name: session
        in: body
        required: true
        schema:
          type: object
          properties:
            session_id:
              type: string
            session_config:
              type: object
    responses:
        201:
            description: The created session
            schema:
                $ref: '#/definitions/Session'
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        db = Control.instance().db
        data = request.json if request.headers["Content-Type"] == "application/json" else request.form.to_dict()

        session_config = SessionConfigDTO()
        session_config.populate_with(data.pop("session_config"))

        session = SessionDTO()
        session.session_id = None
        session.session_config = session_config
        session.populate_with(data)

        result = db.session_store.add(session)

        response = result.to_dict()
        status_code: int = 201

        return jsonify(response), status_code
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"message": e.user_message()}), 400
    except MissingFieldError as e:
        logger.error(f"Missing field error: {e}")
        return jsonify({"message": e.user_message()}), 400
    except ValueError as e:
        logger.error(f"ValueError occurred: {e}")
        return jsonify({"message": "Invalid object"}), 400
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


def _get_number_of_available_clients():
    control = Control.instance()
    result = 0
    for combiner in control.network.get_combiners():
        try:
            nr_active_clients = len(combiner.list_active_clients())
            result = result + int(nr_active_clients)
        except CombinerUnavailableError:
            return 0

    return result


@bp.route("/start", methods=["POST"])
@jwt_auth_required(role="admin")
def start_session():
    """Start a new session.
    param: session_id: The session id to start.
    type: session_id: str
    param: rounds: The number of rounds to run.
    type: rounds: int
    """
    try:
        db = Control.instance().db
        control = Control.instance()
        data = request.json if request.headers["Content-Type"] == "application/json" else request.form.to_dict()
        session_id: str = data.get("session_id")
        rounds: int = data.get("rounds", "")
        round_timeout: int = data.get("round_timeout", None)
        model_name_prefix: str = data.get("model_name_prefix", None)

        if model_name_prefix is None or not isinstance(model_name_prefix, str) or len(model_name_prefix) == 0:
            model_name_prefix = None

        if not session_id or session_id == "":
            return jsonify({"message": "Session ID is required"}), 400

        session = db.session_store.get(session_id)

        session_config = session.session_config
        model_id = session_config.model_id
        min_clients = session_config.clients_required

        if control.state() == ReducerState.monitoring:
            return jsonify({"message": "A session is already running!"}), 400

        if not rounds or not isinstance(rounds, int):
            rounds = session_config.rounds
        nr_available_clients = _get_number_of_available_clients()

        if nr_available_clients < min_clients:
            return jsonify({"message": f"Number of available clients is lower than the required minimum of {min_clients}"}), 400

        model = db.model_store.get(model_id)
        if model is None:
            return jsonify({"message": "Session seed model not found"}), 400

        threading.Thread(target=control.start_session, args=(session_id, rounds, round_timeout, model_name_prefix)).start()

        return jsonify({"message": "Session started"}), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>", methods=["PATCH"])
@jwt_auth_required(role="admin")
def patch_session(id: str):
    """Patch session
    Updates a session based on the provided id. Only the fields that are present in the request will be updated.
    ---
    tags:
        - Sessions
    parameters:
        - name: id
            in: path
            required: true
            type: string
            description: The id or session property of the session
        - name: session
            in: body
            required: true
            type: object
            description: The session data to update
    responses:
        200:
            description: The updated session
            schema:
                $ref: '#/definitions/Session'
        404:
            description: The session was not found
            schema:
                type: object
                properties:
                    message:
                        type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        db = Control.instance().db
        existing_session = db.session_store.get(id)
        if existing_session is None:
            return jsonify({"message": f"Entity with id: {id} not found"}), 404

        data = request.get_json()
        # Remove session_id from the data if it exists
        # since we are editing 'id' otherwise the user could change the id
        data.pop("session_id", None)

        existing_session.patch_with(data, throw_on_extra_keys=False)
        updated_session = db.session_store.update(existing_session)

        response = updated_session.to_dict()
        return jsonify(response), 200

    except EntityNotFound as e:
        logger.error(f"Entity not found: {e}")
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"message": e.user_message()}), 400
    except MissingFieldError as e:
        logger.error(f"Missing field error: {e}")
        return jsonify({"message": e.user_message()}), 400
    except ValueError as e:
        logger.error(f"ValueError occured: {e}")
        return jsonify({"message": "Invalid object"}), 400
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>", methods=["PUT"])
@jwt_auth_required(role="admin")
def put_session(id: str):
    """Put session
    Updates a session based on the provided id. All fields will be updated with the new data.
    ---
    tags:
        - Sessions
    parameters:
        - name: id
            in: path
            required: true
            type: string
            description: The id or session property of the session
        - name: session
            in: body
            required: true
            type: object
            description: The session data to update
    responses:
        200:
            description: The updated session
            schema:
                $ref: '#/definitions/Session'
        404:
            description: The session was not found
            schema:
                type: object
                properties:
                    message:
                        type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        db = Control.instance().db
        session = db.session_store.get(id)
        if session is None:
            return jsonify({"message": f"Entity with id: {id} not found"}), 404

        data = request.get_json()
        data["session_id"] = id
        session.populate_with(data)

        updated_session = db.session_store.update(session)

        response = updated_session.to_dict()
        return jsonify(response), 200

    except EntityNotFound as e:
        logger.error(f"Entity not found: {e}")
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"message": e.user_message()}), 400
    except MissingFieldError as e:
        logger.error(f"Missing field error: {e}")
        return jsonify({"message": e.user_message()}), 400
    except ValueError as e:
        logger.error(f"ValueError occured: {e}")
        return jsonify({"message": "Invalid object"}), 400
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500

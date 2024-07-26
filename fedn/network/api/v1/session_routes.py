import threading

from flask import Blueprint, jsonify, request

from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.shared import control
from fedn.network.api.v1.shared import api_version, get_post_data_to_kwargs, get_typed_list_headers, mdb
from fedn.network.combiner.interfaces import CombinerUnavailableError
from fedn.network.state import ReducerState
from fedn.network.storage.statestore.stores.session_store import SessionStore
from fedn.network.storage.statestore.stores.shared import EntityNotFound

from .model_routes import model_store

bp = Blueprint("session", __name__, url_prefix=f"/api/{api_version}/sessions")

session_store = SessionStore(mdb, "control.sessions")


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
        limit, skip, sort_key, sort_order, _ = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        sessions = session_store.list(limit, skip, sort_key, sort_order, use_typing=False, **kwargs)

        result = sessions["result"]

        response = {"count": sessions["count"], "result": result}

        return jsonify(response), 200
    except Exception:
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
        limit, skip, sort_key, sort_order, _ = get_typed_list_headers(request.headers)
        kwargs = get_post_data_to_kwargs(request)

        sessions = session_store.list(limit, skip, sort_key, sort_order, use_typing=False, **kwargs)

        result = sessions["result"]

        response = {"count": sessions["count"], "result": result}

        return jsonify(response), 200
    except Exception:
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
        kwargs = request.args.to_dict()
        count = session_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception:
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
        kwargs = get_post_data_to_kwargs(request)
        count = session_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception:
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
        session = session_store.get(id, use_typing=False)
        response = session

        return jsonify(response), 200
    except EntityNotFound:
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except Exception:
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
        data = request.json if request.headers["Content-Type"] == "application/json" else request.form.to_dict()
        successful, result = session_store.add(data)
        response = result
        status_code: int = 201 if successful else 400

        return jsonify(response), status_code
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


def _get_number_of_available_clients():
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
        data = request.json if request.headers["Content-Type"] == "application/json" else request.form.to_dict()
        session_id: str = data.get("session_id")
        rounds: int = data.get("rounds", "")
        round_timeout: int = data.get("round_timeout", None)

        if not session_id or session_id == "":
            return jsonify({"message": "Session ID is required"}), 400

        session = session_store.get(session_id, use_typing=False)

        session_config = session["session_config"]
        model_id = session_config["model_id"]
        min_clients = session_config["clients_required"]

        if control.state() == ReducerState.monitoring:
            return jsonify({"message": "A session is already running."})

        if not rounds or not isinstance(rounds, int):
            rounds = session_config["rounds"]
        nr_available_clients = _get_number_of_available_clients()

        if nr_available_clients < min_clients:
            return jsonify({"message": f"Number of available clients is lower than the required minimum of {min_clients}"}), 400

        _ = model_store.get(model_id, use_typing=False)

        threading.Thread(target=control.start_session, args=(session_id, rounds, round_timeout)).start()

        return jsonify({"message": "Session started"}), 200
    except Exception:
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
        session = session_store.get(id, use_typing=False)

        data = request.get_json()
        _id = session["id"]

        # Update the session with the new data
        # Only update the fields that are present in the request
        for key, value in data.items():
            if key in ["_id", "session_id"]:
                continue
            session[key] = value

        success, message = session_store.update(_id, session)

        if success:
            response = session
            return jsonify(response), 200

        return jsonify({"message": f"Failed to update session: {message}"}), 500
    except EntityNotFound:
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except Exception:
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
        session = session_store.get(id, use_typing=False)
        data = request.get_json()
        _id = session["id"]

        success, message = session_store.update(_id, data)

        if success:
            response = session
            return jsonify(response), 200

        return jsonify({"message": f"Failed to update session: {message}"}), 500
    except EntityNotFound:
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500

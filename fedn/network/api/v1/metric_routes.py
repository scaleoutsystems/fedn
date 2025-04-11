from flask import Blueprint, jsonify, request

from fedn.common.log_config import logger
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.shared import metric_store
from fedn.network.api.v1.shared import api_version, get_post_data_to_kwargs, get_typed_list_headers

bp = Blueprint("metric", __name__, url_prefix=f"/api/{api_version}/metrics")


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
def get_metrics():
    """Get metrics
    Retrieves a list of metrics based on the provided parameters.
    By specifying a parameter in the URL, you can filter the metrics based on that parameter,
    and the response will contain only the metrics that match the filter.
    ---
    tags:
        - Metrics
    parameters:
      - name: sender.name
        in: query
        required: false
        type: string
        description: Name of the sender
      - name: sender.role
        in: query
        required: false
        type: string
        description: Role of the sender
      - name: model_id
        in: query
        required: false
        type: string
        description: Model ID associated with the metric
      - name: model_step
        in: query
        required: false
        type: integer
        description: Model step associated with the metric
      - name: round_id
        in: query
        required: false
        type: string
        description: Round ID associated with the metric
      - name: session_id
        in: query
        required: false
        type: string
        description: Session ID associated with the metric
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of metrics to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of metrics to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the metrics by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the metrics in ('asc' or 'desc')
    definitions:
      Metric:
        type: object
        properties:
          metric_id:
            type: string
          key:
            type: string
          value:
            type: number
          timestamp:
            type: string
            format: date-time
          sender:
            type: object
            properties:
              name:
                type: string
              role:
                type: string
          model_id:
            type: string
          model_step:
            type: integer
          round_id:
            type: string
          session_id:
            type: string
    responses:
      200:
        description: A list of metrics and the total count.
        schema:
          type: object
          properties:
            count:
              type: integer
            result:
              type: array
              items:
                $ref: '#/definitions/Metric'
      500:
        description: An error occurred
        schema:
          type: object
          properties:
            message:
              type: string
    """
    try:
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        result = metric_store.list(limit, skip, sort_key, sort_order, **kwargs)
        count = metric_store.count(**kwargs)
        response = {"count": count, "result": [item.to_dict() for item in result]}

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@jwt_auth_required(role="admin")
@bp.route("/list", methods=["POST"])
def list_metrics():
    """Retrieve a list of metrics based on the provided filters and pagination.

    This endpoint allows an admin user to fetch a list of metrics from the metric store.
    The results can be filtered, sorted, and paginated using the request headers and body.
    ---
    tags:
        - Metrics
    parameters:
    - name: limit
        in: header
        required: false
        type: integer
        description: The maximum number of metrics to retrieve
    - name: skip
        in: header
        required: false
        type: integer
        description: The number of metrics to skip
    - name: sort_key
        in: header
        required: false
        type: string
        description: The key to sort the metrics by
    - name: sort_order
        in: header
        required: false
        type: string
        description: The order to sort the metrics in ('asc' or 'desc')
    - name: filters
        in: body
        required: false
        schema:
        type: object
        additionalProperties:
            type: string
        description: Additional filters for querying metrics
    definitions:
    Metric:
        type: object
        properties:
        metric_id:
            type: string
        key:
            type: string
        value:
            type: number
        timestamp:
            type: string
            format: date-time
        sender:
            type: object
            properties:
            name:
                type: string
            role:
                type: string
        model_id:
            type: string
        model_step:
            type: integer
        round_id:
            type: string
        session_id:
            type: string
    responses:
    200:
        description: A list of metrics and the total count.
        schema:
        type: object
        properties:
            count:
            type: integer
            result:
            type: array
            items:
                $ref: '#/definitions/Metric'
    500:
        description: An error occurred
        schema:
        type: object
        properties:
            message:
            type: string
    """
    try:
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = get_post_data_to_kwargs(request)

        result = metric_store.list(limit, skip, sort_key, sort_order, **kwargs)
        count = metric_store.count(**kwargs)
        response = {"count": count, "result": [item.to_dict() for item in result]}

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500

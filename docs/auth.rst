.. _auth-label:

Authentication and Authorization (RBAC)
=============================================
.. warning:: The FEDn RBAC system is an experimental feature and may change in the future.

FEDn supports Role-Based Access Control (RBAC) for controlling access to the FEDn API and gRPC endpoints. The RBAC system is based on JSON Web Tokens (JWT) and is implemented using the `jwt` package. The JWT tokens are used to authenticate users and to control access to the FEDn API.
There are two types of JWT tokens used in the FEDn RBAC system:
- Access tokens: Used to authenticate users and to control access to the FEDn API.
- Refresh tokens: Used to obtain new access tokens when the old ones expire.
 
.. note:: Please note that the FEDn RBAC system is not enabled by default and does not issue JWT tokens. It is used to integrate with external authentication and authorization systems such as FEDn Studio. 

FEDn RBAC system is by default configured with four types of roles:
- `admin`: Has full access to the FEDn API. This role is used to manage the FEDn network using the API client or the FEDn CLI.
- `combiner`: Has access to the /add_combiner endpoint in the API.
- `client`: Has access to the /add_client endpoint in the API and various gRPC endpoint to participate in federated learning sessions.

A full list of the "roles to endpoint" mappings for gRPC can be found in the `fedn/network/grpc/auth.py`. For the API, the mappings are defined using custom decorators defined in `fedn/network/api/auth.py`.

.. note:: The roles are handled by a custom claim in the JWT token called `role`. The claim is used to control access to the FEDn API and gRPC endpoints.

To enable the FEDn RBAC system, you need to set the following environment variables in the controller and combiner:

.. envvar:: FEDN_JWT_SECRET_KEY
    :type: str
    :required: yes
    :default: None
    :description: The secret key used for JWT token encryption.

.. envvar:: FEDN_JWT_ALGORITHM
    :type: str
    :required: no
    :default: "HS256"
    :description: The algorithm used for JWT token encryption.

.. envvar:: FEDN_AUTH_SCHEME
    :type: str
    :required: no
    :default: "Token"
    :description: The authentication scheme used in the FEDn API and gRPC interceptors.

For further fexibility, you can also set the following environment variables:

.. envvar:: FEDN_CUSTOM_URL_PREFIX
    :type: str
    :required: no
    :default: None
    :description: Add a custom URL prefix used in the FEDn API, such as /internal or /v1.

.. envvar:: FEDN_AUTH_WHITELIST_URL
    :type: str
    :required: no
    :default: None
    :description: A URL patterns to the API that should be excluded from the FEDn RBAC system. For example /internal (to enable internal API calls).

.. envvar:: FEDN_JWT_CUSTOM_CLAIM_KEY
    :type: str
    :required: no
    :default: None
    :description: The custom claim key used in the JWT token.

.. envvar:: FEDN_JWT_CUSTOM_CLAIM_VALUE
    :type: str
    :required: no
    :default: None
    :description: The custom claim value used in the JWT token.


For the client you need to set the following environment variables:

.. envvar:: FEDN_JWT_ACCESS_TOKEN
    :type: str
    :required: yes
    :default: None
    :description: The access token used to authenticate the client to the FEDn API.

.. envvar:: FEDN_JWT_REFRESH_TOKEN
    :type: str
    :required: no
    :default: None
    :description: The refresh token used to obtain new access tokens when the old ones expire.

.. envvar:: FEDN_AUTH_SCHEME
    :type: str
    :required: no
    :default: "Token"
    :description: The authentication scheme used in the FEDn API and gRPC interceptors.

You can also use `--token` flags in the FEDn CLI to set the access token.
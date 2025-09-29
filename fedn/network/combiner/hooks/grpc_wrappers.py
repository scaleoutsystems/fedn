import time

import grpc

from fedn.common.log_config import logger


def safe_unary(func_name, default_resp_factory):
    def decorator(fn):
        def wrapper(self, request, context):
            try:
                return fn(self, request, context)
            except Exception as e:
                self._retire_and_log(func_name, e)
                # Option A: return a valid default payload (keeps channel healthy)
                return default_resp_factory()

        return wrapper

    return decorator


def safe_streaming(func_name):
    def decorator(fn):
        def wrapper(self, request, context):
            try:
                yield from fn(self, request, context)
            except Exception as e:
                self.client_updates = {}
                self._retire_and_log(func_name, e)
                # Option B for streaming: signal an RPC error the client understands
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(f"{func_name} failed; sender should use local fallback.")
                return

        return wrapper

    return decorator


def call_with_fallback(name, fn, *, retries=2, base_sleep=0.25, fallback_fn=None):
    for i in range(retries + 1):
        try:
            return fn()
        except grpc.RpcError as e:
            code = e.code()
            if code in (grpc.StatusCode.FAILED_PRECONDITION, grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                logger.warning(f"{name} rpc failed with {code.name}: {e.details()}; attempt {i + 1}/{retries}")
                if i < retries:
                    time.sleep(base_sleep * (2**i))
                    continue
            break
        except Exception as e:
            logger.exception(f"{name} unexpected error: {e}")
            break
    if fallback_fn:
        logger.info(f"{name}: using local fallback")
        return fallback_fn()
    raise RuntimeError(f"{name} failed and no fallback provided")

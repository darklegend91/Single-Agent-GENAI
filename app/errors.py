"""Client-safe controlled errors; original provider exceptions stay internal."""


class AppError(Exception):
    status_code = 500
    code = "internal_error"
    message = "The request could not be completed."

    def __init__(self) -> None:
        super().__init__(self.message)


class AuthenticationError(AppError):
    status_code = 401
    code = "authentication_required"
    message = "Valid authentication is required."


class AuthorizationError(AppError):
    status_code = 403
    code = "access_denied"
    message = "Access to this resource is denied."


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
    message = "The requested resource was not found."


class CustomerNotFoundError(NotFoundError):
    code = "customer_not_found"
    message = "This customer ID is not present in the active indexed dataset."


class ConflictError(AppError):
    status_code = 409
    code = "conflict"
    message = "An operation is already in progress."


class ValidationFailure(AppError):
    status_code = 422
    code = "validation_failed"
    message = "The supplied data could not be validated."


class RateLimitError(AppError):
    status_code = 429
    code = "rate_limited"
    message = "The request limit has been reached."


class DependencyError(AppError):
    status_code = 503
    code = "dependency_unavailable"
    message = "A required service is unavailable."


class RequestTimeoutError(AppError):
    status_code = 504
    code = "request_timeout"
    message = "The request exceeded its time limit."

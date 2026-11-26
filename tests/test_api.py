from mess_waste.api import app


def test_health_endpoint_returns_ok():
    health_route = next(route for route in app.routes if route.path == "/health")
    assert health_route.endpoint() == {"status": "ok"}

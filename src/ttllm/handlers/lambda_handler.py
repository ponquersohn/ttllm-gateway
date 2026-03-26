"""AWS Lambda handler via Mangum."""

from mangum import Mangum

from ttllm.api.app import create_app

app = create_app()
handler = Mangum(app, lifespan="off")

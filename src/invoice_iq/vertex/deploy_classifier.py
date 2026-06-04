"""Build a gated Vertex AI deployment command plan.

This module never executes `gcloud`. It prints the commands that a human can
review and run after the Phase 7 cost gate is approved.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class VertexDeployPlan:
    """A no-execute command plan for deploying the API image to Vertex."""

    project_id: str
    region: str
    image_uri: str
    model_display_name: str
    endpoint_display_name: str

    def commands(self) -> list[str]:
        return [
            "gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com",
            f"docker push {self.image_uri}",
            (
                "MODEL_ID=$(gcloud ai models upload "
                f"--project={self.project_id} --region={self.region} "
                f"--display-name={self.model_display_name} "
                f"--container-image-uri={self.image_uri} "
                "--container-predict-route=/predict "
                "--container-health-route=/health "
                "--container-ports=8000 "
                "--format='value(name)')"
            ),
            (
                "ENDPOINT_ID=$(gcloud ai endpoints create "
                f"--project={self.project_id} --region={self.region} "
                f"--display-name={self.endpoint_display_name} "
                "--format='value(name)')"
            ),
            (
                "gcloud ai endpoints deploy-model $ENDPOINT_ID "
                f"--project={self.project_id} --region={self.region} "
                "--model=$MODEL_ID "
                "--display-name=invoice-iq-classifier "
                "--traffic-split=0=100 "
                "--machine-type=n1-standard-2 "
                "--min-replica-count=1 --max-replica-count=1"
            ),
        ]


def build_plan(
    *,
    project_id: str,
    region: str,
    image_uri: str,
    model_display_name: str = "invoice-iq-classifier",
    endpoint_display_name: str = "invoice-iq-classifier-endpoint",
) -> VertexDeployPlan:
    """Create a no-execute deploy plan."""
    return VertexDeployPlan(
        project_id=project_id,
        region=region,
        image_uri=image_uri,
        model_display_name=model_display_name,
        endpoint_display_name=endpoint_display_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print Vertex AI deployment commands.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--image-uri", required=True)
    args = parser.parse_args()

    plan = build_plan(project_id=args.project, region=args.region, image_uri=args.image_uri)
    print("# Review these commands. They are not executed by this script.")
    print("# Cost gate: running them can create billable Vertex AI resources.")
    for command in plan.commands():
        print(command)


if __name__ == "__main__":
    main()

import os
from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
)
from constructs import Construct


class BackendStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        **kwargs,
    ):
        super().__init__(scope, construct_id, **kwargs)

        vpc = ec2.Vpc(
            self,
            "BackendVpc",
            max_azs=1,
            nat_gateways=0,
        )

        security_group = ec2.SecurityGroup(
            self,
            "BackendSecurityGroup",
            vpc=vpc,
            description="Security group for FastAPI backend",
            allow_all_outbound=True,
        )

        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(8000),
            "FastAPI",
        )

        instance = ec2.Instance(
            self,
            "BackendInstance",
            vpc=vpc,
            instance_type=ec2.InstanceType("t3.small"),
            machine_image=(
                ec2.MachineImage.latest_amazon_linux2023()
            ),
            security_group=security_group,
        )

        instance.add_user_data(
            "dnf update -y",
            "dnf install -y docker",
            "systemctl enable docker",
            "systemctl start docker",
            "usermod -a -G docker ec2-user",
        )


if __name__ == "__main__":
    import aws_cdk as cdk
    
    app = cdk.App()
    BackendStack(
        app,
        "BackendStack",
        env=cdk.Environment(
            account=os.getenv("CDK_DEFAULT_ACCOUNT"),
            region=os.getenv("CDK_DEFAULT_REGION"),
        ),
    )

    app.synth()

import os
import aws_cdk as cdk

from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_iam as iam,
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


        role = iam.Role(
            self,
            "BackendInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSSMManagedInstanceCore"
                )
            ],
        )

        repository = ecr.Repository(
            self,
            "BackendRepository",
            repository_name="factored-backend",
        )

        github_provider = iam.OpenIdConnectProvider(
            self,
            "GitHubOIDCProvider",
            url="https://token.actions.githubusercontent.com",
            client_ids=["sts.amazonaws.com"],
        )

        github_role = iam.Role(
            self,
            "GitHubActionsRole",
            assumed_by=iam.WebIdentityPrincipal(
                github_provider.open_id_connect_provider_arn,
                conditions={
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                    },
                    "StringLike": {
                        "token.actions.githubusercontent.com:sub":
                        "repo:/mavino4/DatathonFactored:*",
                    },
                },
            ),
        )

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
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PUBLIC,
            ),
            associate_public_ip_address=True,
            instance_type=ec2.InstanceType("t3.small"),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(),
            security_group=security_group,
            role=role,
        )

        instance.add_user_data(
            "dnf update -y",
            "dnf install -y docker",
            "systemctl enable docker",
            "systemctl start docker",
            "systemctl enable amazon-ssm-agent",
            "systemctl start amazon-ssm-agent",
            "usermod -a -G docker ec2-user",
        )

        cdk.CfnOutput(
            self,
            "InstancePublicIp",
            value=instance.instance_public_dns_name,
        )


if __name__ == "__main__":
    
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

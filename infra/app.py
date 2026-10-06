import os
import aws_cdk as cdk

from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_ecr as ecr,
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
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonEC2ContainerRegistryReadOnly"
                ),
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
                        # Repos created after 2026-07-15 put immutable owner/repo
                        # ids in the OIDC sub claim, plus the current repo name:
                        # repo:OWNER@OWNER_ID/REPO@REPO_ID:ref:...
                        "token.actions.githubusercontent.com:sub":
                        "repo:mavino4@33271590/factored-hackathon-2026-hooked-on-data@1391763945:*",
                    },
                },
            ),
        )

        github_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"],
            )
        )
        github_role.add_to_policy(
            iam.PolicyStatement(
                actions=["cloudformation:DescribeStacks"],
                resources=[self.stack_id],
            )
        )
        repository.grant_pull_push(github_role)

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
        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(3000),
            "Langfuse",
        )
        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(443),
            "HTTPS so the browser allows the microphone",
        )

        instance = ec2.Instance(
            self,
            "BackendInstance",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PUBLIC,
            ),
            associate_public_ip_address=True,
            instance_type=ec2.InstanceType("t3.large"),
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

        github_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ssm:SendCommand"],
                resources=[
                    f"arn:aws:ssm:{self.region}::document/AWS-RunShellScript",
                    self.format_arn(
                        service="ec2",
                        resource="instance",
                        resource_name=instance.instance_id,
                    ),
                ],
            )
        )
        github_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ssm:GetCommandInvocation"],
                resources=["*"],
            )
        )

        cdk.CfnOutput(
            self,
            "InstanceId",
            value=instance.instance_id,
        )

        cdk.CfnOutput(
            self,
            "InstancePublicIp",
            value=instance.instance_public_ip,
        )

        cdk.CfnOutput(
            self,
            "BackendUrlIp",
            value="http://" + instance.instance_public_ip + ":8000/",
            description="Application URL using the public IP",
        )

        cdk.CfnOutput(
            self,
            "BackendUrlHost",
            value="http://" + instance.instance_public_dns_name + ":8000/",
            description="Application URL using the public DNS name",
        )

        cdk.CfnOutput(
            self,
            "BackendHttpsUrlIp",
            value="https://" + instance.instance_public_ip + "/",
            description="Application HTTPS URL using the public IP",
        )

        cdk.CfnOutput(
            self,
            "BackendHttpsUrlHost",
            value="https://" + instance.instance_public_dns_name + "/",
            description="Application HTTPS URL using the public DNS name",
        )

        cdk.CfnOutput(
            self,
            "LangfuseUrlIp",
            value="http://" + instance.instance_public_ip + ":3000/",
            description="Langfuse URL using the public IP",
        )

        cdk.CfnOutput(
            self,
            "LangfuseUrlHost",
            value="http://" + instance.instance_public_dns_name + ":3000/",
            description="Langfuse URL using the public DNS name",
        )

        cdk.CfnOutput(
            self,
            "GitHubActionsRoleArn",
            value=github_role.role_arn,
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

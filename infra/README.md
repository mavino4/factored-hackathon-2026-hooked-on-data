# AWS infra

```
npm install -g aws-cdk
uv sync
```

## Bootstrap command

```
cdk bootstrap aws://$(aws sts get-caller-identity --query Account --output text)/$(aws configure get region)
```


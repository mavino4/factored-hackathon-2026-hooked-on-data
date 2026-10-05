# AWS infra

```
npm install -g aws-cdk
uv sync
```

## Bootstrap command

```
cdk bootstrap aws://$(aws sts get-caller-identity --query Account --output text)/$(aws configure get region)
```

## Install the Session Manager Plugin

```
curl "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/ubuntu_64bit/session-manager-plugin.deb" \
  -o session-manager-plugin.deb
sudo dpkg -i session-manager-plugin.deb
```

unset AWS_SECRET_ACCESS_KEY
unset AWS_SECRET_KEY
unset AWS_SESSION_TOKEN
unset AWS_PROFILE
# Replace the variables with your own values
ROLE_ARN=arn:aws:iam::320425221866:role/GreenlandDefaultJobRole
PROFILE='Profile_IML_CV_GEN_AI'
# REGION=<region>
# Assume the role
TEMP_CREDS=$(aws sts assume-role --role-arn "$ROLE_ARN" --role-session-name "temp-session-1" --output json)
echo $TEMP_CREDS
# Extract the necessary information from the response
ACCESS_KEY=$(echo $TEMP_CREDS | jq -r .Credentials.AccessKeyId)
SECRET_KEY=$(echo $TEMP_CREDS | jq -r .Credentials.SecretAccessKey)
SESSION_TOKEN=$(echo $TEMP_CREDS | jq -r .Credentials.SessionToken)
# Put the information into the AWS CLI credentials file
aws configure set aws_access_key_id "$ACCESS_KEY" --profile "$PROFILE"
aws configure set aws_secret_access_key "$SECRET_KEY" --profile "$PROFILE"
aws configure set aws_session_token "$SESSION_TOKEN" --profile "$PROFILE"
# aws configure set region "$REGION" --profile "$PROFILE"
# Export profile to environment 
# NOTE: This line must be manually executed after this script finished.
export AWS_PROFILE='Profile_IML_CV_GEN_AI'
# Setup codecommit
git config --global credential.helper '!aws codecommit credential-helper $@'
git config --global credential.UseHttpPath true

